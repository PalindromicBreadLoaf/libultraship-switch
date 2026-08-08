#include "fast/backends/gfx_shader_cache.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "ship/Context.h"
#include "ship/resource/ResourceManager.h"
#include "ship/resource/File.h"
#include "libultraship/bridge/consolevariablebridge.h"
#include "spdlog/spdlog.h"

namespace Fast {

/*
 * Store layout. Little-endian, append-only, header once at the front:
 *
 *   off  size  field
 *   0    8     magic "GDXSHC\0\1"
 *   8    4     backend tag (SHADER_CACHE_TAG_*)
 *   12   8     fingerprint (build hash ^ backend fingerprint)
 *   20   4     reserved, zero
 *   24   ..    entries until EOF
 *
 * entry:
 *   0    8     shader_id0
 *   8    8     shader_id1
 *   16   4     flags (SHADER_CACHE_FLAG_*)
 *   20   4     payload size
 *   24   ..    payload
 *
 * Append-only so each compile is durable the instant it happens: no shutdown hook to forget,
 * and a crash keeps the session's work. ParseImage's length check drops a torn final entry, at
 * the cost of recompiling that one variant.
 */
static const uint8_t kMagic[8] = { 'G', 'D', 'X', 'S', 'H', 'C', 0x00, 0x01 };
static constexpr size_t kHeaderSize = 24;
static constexpr size_t kEntryHeaderSize = 24;

/** @brief Reject a payload this large as corruption rather than allocating for it. */
static constexpr uint32_t kMaxPayloadSize = 4u * 1024u * 1024u;

#ifndef GDX_SHADER_CACHE_FINGERPRINT
// No configure-time hash reached this translation unit. Refuse to cache rather than cache
// under an unknown fingerprint: a wrong blob costs far more to diagnose than a recompile.
#define GDX_SHADER_CACHE_FINGERPRINT 0ull
#endif

static uint32_t ReadU32(const uint8_t* p) {
    uint32_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static uint64_t ReadU64(const uint8_t* p) {
    uint64_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

bool ShaderCacheUserEnabled() {
    // Env wins for a single run and is never written back; the CVar is the persisted preference.
    // Same precedence as the port's logging gates.
    const char* env = getenv("GDX_SHADER_CACHE");
    if (env != nullptr && env[0] != '\0') {
        return strcmp(env, "0") != 0;
    }
    return CVarGetInteger("gDevTools.ShaderCache", 1) != 0;
}

/** @brief True when this image's header matches what this build would write. */
static bool HeaderMatches(const uint8_t* data, size_t size, uint32_t tag, uint64_t fingerprint) {
    if (data == nullptr || size < kHeaderSize) {
        return false;
    }
    if (memcmp(data, kMagic, sizeof(kMagic)) != 0) {
        return false;
    }
    return ReadU32(data + 8) == tag && ReadU64(data + 12) == fingerprint;
}

size_t ShaderBlobCache::ParseImage(const uint8_t* data, size_t size, bool fromSeed) {
    if (!HeaderMatches(data, size, mBackendTag, mFingerprint)) {
        return 0;
    }

    size_t accepted = 0;
    size_t off = kHeaderSize;
    while (off + kEntryHeaderSize <= size) {
        const uint64_t id0 = ReadU64(data + off);
        const uint64_t id1 = ReadU64(data + off + 8);
        const uint32_t flags = ReadU32(data + off + 16);
        const uint32_t payloadSize = ReadU32(data + off + 20);

        if (payloadSize == 0 || payloadSize > kMaxPayloadSize) {
            break; // corrupt length: stop, keep everything parsed so far
        }
        if (off + kEntryHeaderSize + payloadSize > size) {
            break; // torn final entry (see the append-only note above)
        }

        const uint8_t* payload = data + off + kEntryHeaderSize;
        // emplace, not insert_or_assign: seed and sidecar copies of a variant hold the same
        // bytes, and the seed was read first, so first-writer-wins costs nothing.
        mEntries.emplace(Key{ id0, id1, flags }, std::vector<uint8_t>(payload, payload + payloadSize));

        ++accepted;
        off += kEntryHeaderSize + payloadSize;
    }

    if (fromSeed) {
        mSeedEntries += accepted;
    } else {
        mSidecarEntries += accepted;
    }
    return accepted;
}

void ShaderBlobCache::OpenSidecar(const std::string& path) {
    mSidecarPath = path;

    std::vector<uint8_t> image;
    bool headerOk = false;
    if (FILE* rf = fopen(path.c_str(), "rb")) {
        fseek(rf, 0, SEEK_END);
        const long len = ftell(rf);
        if (len > 0) {
            fseek(rf, 0, SEEK_SET);
            image.resize((size_t)len);
            if (fread(image.data(), 1, image.size(), rf) != image.size()) {
                image.clear();
            }
        }
        fclose(rf);
        headerOk = HeaderMatches(image.data(), image.size(), mBackendTag, mFingerprint);
    }

    if (headerOk) {
        ParseImage(image.data(), image.size(), false);
        mSidecarHandle = fopen(path.c_str(), "ab");
    } else {
        // Absent, empty, or written by a build whose shader generators have since changed.
        // Truncate and re-header rather than grow a file nothing can read.
        FILE* wf = fopen(path.c_str(), "wb");
        if (wf != nullptr) {
            uint8_t header[kHeaderSize];
            memset(header, 0, sizeof(header));
            memcpy(header, kMagic, sizeof(kMagic));
            memcpy(header + 8, &mBackendTag, sizeof(mBackendTag));
            memcpy(header + 12, &mFingerprint, sizeof(mFingerprint));
            if (fwrite(header, 1, sizeof(header), wf) == sizeof(header)) {
                fflush(wf);
                mSidecarHandle = wf;
            } else {
                fclose(wf);
            }
        }
    }

    mSidecarWritable = mSidecarHandle != nullptr;
    if (!mSidecarWritable) {
        // Packaged installs can put the program directory on read-only media. Not fatal -- the
        // shipped seed still works and the run behaves as it did before the cache existed.
        SPDLOG_WARN("[shader-cache] sidecar not writable at '{}'; cache is read-only this run", path);
    }
}

void ShaderBlobCache::Init(uint32_t backendTag, uint64_t backendFingerprint, const char* seedResourcePath,
                           const char* sidecarFileName) {
    mEntries.clear();
    mSeedEntries = 0;
    mSidecarEntries = 0;
    mEnabled = false;

    if (mSidecarHandle != nullptr) {
        fclose((FILE*)mSidecarHandle);
        mSidecarHandle = nullptr;
    }

    const uint64_t buildFingerprint = (uint64_t)(GDX_SHADER_CACHE_FINGERPRINT);
    if (buildFingerprint == 0ull) {
        SPDLOG_WARN("[shader-cache] disabled: this build carries no generator fingerprint "
                    "(GDX_SHADER_CACHE_FINGERPRINT unset in CMake)");
        return;
    }
    if (!ShaderCacheUserEnabled()) {
        // WARN, not INFO, here and on the ready line below: Release builds initialise the logger
        // at spdlog::level::warn (Context.h InitLogging default), so INFO is discarded entirely.
        SPDLOG_WARN("[shader-cache] disabled by gDevTools.ShaderCache / GDX_SHADER_CACHE");
        return;
    }

    mBackendTag = backendTag;
    mFingerprint = buildFingerprint ^ backendFingerprint;
    mEnabled = true;

    // Seed before sidecar: it is read-only and authoritative, and the emplace-wins-first rule
    // above depends on this order.
    if (seedResourcePath != nullptr) {
        try {
            auto file = Ship::Context::GetInstance()->GetResourceManager()->LoadFileProcess(
                std::string(seedResourcePath));
            if (file != nullptr && file->Buffer != nullptr) {
                const size_t trueSize = file->TrueSize != 0 ? file->TrueSize : file->Buffer->size();
                const size_t available = file->Buffer->size() - file->BufferOffset;
                const size_t usable = trueSize < available ? trueSize : available;
                const uint8_t* base = (const uint8_t*)file->Buffer->data() + file->BufferOffset;
                const size_t n = ParseImage(base, usable, true);
                if (n == 0 && usable >= kHeaderSize) {
                    // Present but unusable: almost always a seed recorded before a shader
                    // generator was edited. Worth naming; the symptom is only "first run stutters".
                    SPDLOG_WARN("[shader-cache] seed '{}' rejected (fingerprint or backend mismatch)",
                                seedResourcePath);
                }
            }
        } catch (const std::exception& e) {
            SPDLOG_WARN("[shader-cache] seed '{}' unreadable: {}", seedResourcePath, e.what());
        } catch (...) {
            SPDLOG_WARN("[shader-cache] seed '{}' unreadable", seedResourcePath);
        }
    }

    if (sidecarFileName != nullptr) {
        OpenSidecar(Ship::Context::GetPathRelativeToAppDirectory(sidecarFileName));
    }

    SPDLOG_WARN("[shader-cache] ready: {} entries ({} seed, {} sidecar), fingerprint {:016X}, sidecar {}",
                mEntries.size(), mSeedEntries, mSidecarEntries, mFingerprint,
                mSidecarWritable ? "writable" : "read-only");
}

const std::vector<uint8_t>* ShaderBlobCache::Lookup(uint64_t shaderId0, uint64_t shaderId1, uint32_t flags) {
    if (!mEnabled) {
        return nullptr;
    }
    auto it = mEntries.find(Key{ shaderId0, shaderId1, flags });
    return it == mEntries.end() ? nullptr : &it->second;
}

void ShaderBlobCache::Store(uint64_t shaderId0, uint64_t shaderId1, uint32_t flags, const void* payload,
                            size_t size) {
    if (!mEnabled || payload == nullptr || size == 0 || size > kMaxPayloadSize) {
        return;
    }

    const uint8_t* bytes = (const uint8_t*)payload;
    const Key key{ shaderId0, shaderId1, flags };
    if (!mEntries.emplace(key, std::vector<uint8_t>(bytes, bytes + size)).second) {
        return; // already known in memory; nothing to append
    }

    if (mSidecarHandle == nullptr) {
        return;
    }

    uint8_t header[kEntryHeaderSize];
    const uint32_t payloadSize = (uint32_t)size;
    memcpy(header, &shaderId0, sizeof(shaderId0));
    memcpy(header + 8, &shaderId1, sizeof(shaderId1));
    memcpy(header + 16, &flags, sizeof(flags));
    memcpy(header + 20, &payloadSize, sizeof(payloadSize));

    FILE* f = (FILE*)mSidecarHandle;
    const bool wroteHeader = fwrite(header, 1, sizeof(header), f) == sizeof(header);
    const bool wrotePayload = wroteHeader && fwrite(bytes, 1, size, f) == size;
    if (wrotePayload) {
        // Flush per entry, not at exit: compiles are rare (tens per install), and there is no
        // shutdown hook that could be trusted to run.
        fflush(f);
    } else {
        SPDLOG_WARN("[shader-cache] append failed for id0={:016X} id1={:016X}; disabling writes",
                    shaderId0, shaderId1);
        fclose(f);
        mSidecarHandle = nullptr;
        mSidecarWritable = false;
    }
}

} // namespace Fast
