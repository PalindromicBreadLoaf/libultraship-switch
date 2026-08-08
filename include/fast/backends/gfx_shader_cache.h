#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace Fast {

/*
 * Persistent store for backend-compiled shader payloads, keyed by the Fast3D combiner variant.
 *
 * Both backends compile an unseen (shader_id0, shader_id1) pair synchronously from inside the
 * draw call. On D3D11 that is 9-15ms per variant, and they arrive in bursts: eleven inside one
 * tick produced a 181ms stall. The variant set is finite (~37 a session), so this turns a
 * per-launch warm-up cost into a per-install one.
 *
 * The payload is opaque here -- DXBC blobs on D3D11, a linked program binary on OpenGL.
 * Everything else a ShaderProgram needs derives from cc_features; numFloats is the exception,
 * since it falls out of source generation and a cache hit skips that, so it rides in the payload.
 *
 * A blob is only valid for the source its generator emits today, so every store carries
 * GDX_SHADER_CACHE_FINGERPRINT (see src/fast/CMakeLists.txt) and a mismatch discards the file.
 * A build without a fingerprint refuses to cache at all rather than guess.
 *
 * Entries come from a read-only seed in the port archive and a writable sidecar next to the
 * executable. GL program binaries are vendor/GPU/driver specific, so only the sidecar can work
 * there. A miss on both is not an error.
 *
 * Render thread only (Init and the CreateAndLoadNewShader miss path); there is no lock.
 */

/** @brief Backend tag written into the store header; a mismatch rejects the whole file. */
enum ShaderCacheBackendTag : uint32_t {
    SHADER_CACHE_TAG_D3D11 = 0x31314433u, // '3D11'
    SHADER_CACHE_TAG_OPENGL = 0x204C474Fu // 'OGL '
};

/** @brief Variant bits that change generated shader source but are not part of the shader id. */
enum ShaderCacheFlags : uint32_t {
    SHADER_CACHE_FLAG_THREE_POINT = 1u << 0,
    SHADER_CACHE_FLAG_SRGB = 1u << 1
};

class ShaderBlobCache {
  public:
    /**
     * @brief Open the store: load the shipped seed, then the writable sidecar.
     *
     * Safe to call on a build with no fingerprint, with no archive seed, or with an unwritable
     * program directory -- each degrades to "compile as before" and logs the reason once.
     *
     * @param backendTag        Identifies the payload dialect (DXBC vs GL program binary).
     * @param backendFingerprint Extra bits the backend must agree on beyond the build hash. On
     *                          OpenGL this MUST fold in GL_VENDOR/GL_RENDERER/GL_VERSION,
     *                          because program binaries are not portable across drivers.
     * @param seedResourcePath  Path of the seed inside the archive, or nullptr for none.
     * @param sidecarFileName   Sidecar file name, resolved next to the executable.
     */
    void Init(uint32_t backendTag, uint64_t backendFingerprint, const char* seedResourcePath,
              const char* sidecarFileName);

    /** @brief Payload for this variant, or nullptr on a miss. Pointer stays valid until Init. */
    const std::vector<uint8_t>* Lookup(uint64_t shaderId0, uint64_t shaderId1, uint32_t flags);

    /** @brief Record a freshly compiled payload and append it to the sidecar. */
    void Store(uint64_t shaderId0, uint64_t shaderId1, uint32_t flags, const void* payload, size_t size);

    /** @brief False when the cache is inert: no fingerprint, or disabled by CVar/env. */
    bool Enabled() const {
        return mEnabled;
    }

  private:
    struct Key {
        uint64_t Id0;
        uint64_t Id1;
        uint32_t Flags;

        bool operator==(const Key& other) const {
            return Id0 == other.Id0 && Id1 == other.Id1 && Flags == other.Flags;
        }
    };

    struct KeyHash {
        size_t operator()(const Key& k) const {
            // The ids are already well-mixed 64-bit combiner encodings; fold them with the
            // 64-bit FNV prime so the low bits of flags cannot collapse distinct variants.
            uint64_t h = k.Id0 ^ (k.Id1 * 1099511628211ull) ^ ((uint64_t)k.Flags << 56);
            h ^= h >> 33;
            h *= 0xFF51AFD7ED558CCDull;
            h ^= h >> 33;
            return (size_t)h;
        }
    };

    /** @brief Parse a store image into mEntries. Returns entries accepted. */
    size_t ParseImage(const uint8_t* data, size_t size, bool fromSeed);

    /** @brief Open (or re-create) the sidecar so Store can append to it. */
    void OpenSidecar(const std::string& path);

    std::unordered_map<Key, std::vector<uint8_t>, KeyHash> mEntries;
    std::string mSidecarPath;
    void* mSidecarHandle = nullptr; // FILE*, kept opaque so callers need no <cstdio>
    uint32_t mBackendTag = 0;
    uint64_t mFingerprint = 0;
    bool mEnabled = false;
    bool mSidecarWritable = false;

    // Boot summary only. Hit/miss needs no counter: every miss already prints a compile line.
    size_t mSeedEntries = 0;
    size_t mSidecarEntries = 0;
};

/**
 * @brief True unless the user turned the cache off.
 *
 * Persisted CVar, with GDX_SHADER_CACHE overriding it for a single run (the same gate convention
 * as port/port_log.h). Exposed so a backend can skip Init rather than build a disabled store.
 */
bool ShaderCacheUserEnabled();

} // namespace Fast
