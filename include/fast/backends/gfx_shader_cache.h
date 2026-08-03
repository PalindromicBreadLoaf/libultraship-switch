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
 * WHY THIS EXISTS. Both backends build a shader from source the first time an unseen
 * (shader_id0, shader_id1) pair is drawn, synchronously, from inside the draw call. Measured on
 * F-Zero X the D3D11 runtime HLSL compile costs 9-15ms per variant, and variants arrive in
 * bursts as a venue introduces new materials: eleven compiles landed inside a single tick and
 * produced a 181ms stall -- 5.5fps instantaneous -- with game logic at 9ms. Across a whole
 * session it is 37 variants and 410ms, then nothing: a pure warm-up cost. Paying it once per
 * install instead of once per launch removes the stalls without touching the frame pacer.
 *
 * WHAT IS STORED. An opaque per-variant payload the backend serializes for itself: DXBC vertex
 * and pixel bytecode on D3D11, a linked program binary on OpenGL. This class never interprets
 * it. Everything else a ShaderProgram needs (input layout, blend state, attribute and uniform
 * locations) is derived from cc_features or queried from the linked program, so it does not
 * need storing -- with one exception per backend, numFloats, which is a side effect of source
 * generation rather than a function of cc_features and therefore has to ride along inside the
 * payload.
 *
 * WHY A FINGERPRINT. A stored blob is only valid for the exact source text its generator would
 * emit today. Edit an emitter or a .glsl template and every stored blob becomes silently wrong:
 * wrong pixels, or a driver rejecting the binary, on a user's machine, with nothing pointing
 * back here. GDX_SHADER_CACHE_FINGERPRINT is a configure-time hash of those generator inputs
 * (see libultraship/src/fast/CMakeLists.txt), so a mismatched store is discarded automatically.
 * A build that did not receive one refuses to use the cache at all rather than guess.
 *
 * TWO SOURCES, ONE FORMAT. Entries come from a read-only seed shipped inside the port archive
 * and from a writable sidecar next to the executable. The seed makes a fresh install clean on
 * its very first run; the sidecar covers whatever the seed missed and is the only mechanism
 * available on OpenGL, whose program binaries are specific to the vendor, GPU and driver version
 * and therefore cannot be shipped to anyone. A miss on both is not an error -- it compiles as it
 * always did and records the result.
 *
 * THREADING. Touched only from the render thread (Init and the CreateAndLoadNewShader miss
 * path), so there is no lock. Do not call it from anywhere else without adding one.
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

    // Reported in the one-line boot summary. A run's hit/miss split needs no separate counter:
    // every miss prints a [shader-compile] line, so "entries held at boot, and no compile lines
    // afterwards" is the whole story a bug report needs.
    size_t mSeedEntries = 0;
    size_t mSidecarEntries = 0;
};

/**
 * @brief True unless the user turned the cache off.
 *
 * Reads the persisted CVar and lets GDX_SHADER_CACHE override it for a single run, matching the
 * Bucket-D gate convention the port uses elsewhere (see port/port_log.h). Exposed so a backend
 * can skip Init entirely rather than construct a disabled store.
 */
bool ShaderCacheUserEnabled();

} // namespace Fast
