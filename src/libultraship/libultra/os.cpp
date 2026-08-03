#include "libultraship/libultraship.h"
#include "libultraship/bridge/consolevariablebridge.h" /* CVarGetInteger: live audio low-pass cutoff */
#include <SDL2/SDL.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ratio>
#include <vector>

// Establish a chrono duration for the N64 46.875MHz clock rate
typedef std::ratio<3000, 64> n64ClockRatio;
typedef std::ratio_divide<std::micro, n64ClockRatio> n64CycleRate;
typedef std::chrono::duration<long long, n64CycleRate> n64CycleRateDuration;

extern "C" {
uint8_t __osMaxControllers = MAXCONTROLLERS;
uint64_t __osCurrentTime = 0;

struct GdxDecompOSIoMesg {
    OSIoMesgHdr hdr;
    void* dramAddr;
    uint32_t devAddr;
    uint32_t size;
    OSPiHandle* piHandle;
};

/* Whole-file cartridge ROM image owned by the port layer (raw .z64 byte
   stream, i.e. the exact cart byte order). Resolved when the executable
   links this static library. */
extern uint8_t* gdx_rom_buffer;
extern size_t gdx_rom_size;

/* Single byte-source shim (port/gdx_segment_source.c).
   Declared here without its header because this file compiles in the
   libultraship target, which has no include path onto port/ -- the same
   cross-module extern pattern used for gdx_audio_thread_active below. Sources a
   cart read archive-first via the shared segment_blob+audio_blob containment
   table, raw-ROM fallback (byte-identical to a direct gdx_rom_buffer+off copy);
   returns 1 on success (either source), 0 only when no source can satisfy the
   read (ROM absent / out of bounds), leaving dst untouched. */
int GdxSegmentSourceRead(uint32_t romBase, uint32_t size, void* dst);

int32_t osContInit(OSMesgQueue* mq, uint8_t* controllerBits, OSContStatus* status) {
    *controllerBits = 0;
    status->status |= 1;

    std::string controllerDb = Ship::Context::LocateFileAcrossAppDirs("gamecontrollerdb.txt");
    int mappingsAdded = SDL_GameControllerAddMappingsFromFile(controllerDb.c_str());
    if (mappingsAdded >= 0) {
        SPDLOG_INFO("Added SDL game controllers from \"{}\" ({})", controllerDb, mappingsAdded);
    } else {
        SPDLOG_ERROR("Failed add SDL game controller mappings from \"{}\" ({})", controllerDb, SDL_GetError());
    }

    SDL_SetHint(SDL_HINT_JOYSTICK_THREAD, "1");
    if (SDL_Init(SDL_INIT_GAMECONTROLLER) != 0) {
        SPDLOG_ERROR("Failed to initialize SDL game controllers ({})", SDL_GetError());
        exit(EXIT_FAILURE);
    }

    Ship::Context::GetInstance()->GetControlDeck()->Init(controllerBits);

    return 0;
}

int32_t osContStartReadData(OSMesgQueue* mesg) {
    return 0;
}

void osContGetReadData(OSContPad* pad) {
    memset(pad, 0, sizeof(OSContPad) * __osMaxControllers);

    Ship::Context::GetInstance()->GetControlDeck()->WriteToPad(pad);
}

void osSetTime(OSTime time) {
    __osCurrentTime =
        std::chrono::duration_cast<n64CycleRateDuration>(std::chrono::steady_clock::now().time_since_epoch()).count() +
        time;
}

// Returns the OS time matching the N64 46.875MHz cycle rate
uint64_t osGetTime() {
    return std::chrono::duration_cast<n64CycleRateDuration>(std::chrono::steady_clock::now().time_since_epoch())
               .count() -
           __osCurrentTime;
}

// Returns the CPU clock count matching the N64 46.875Mhz cycle rate
uint32_t osGetCount() {
    return std::chrono::duration_cast<n64CycleRateDuration>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

OSPiHandle* osCartRomInit() {
    static OSPiHandle sCartRomHandle = {};
    sCartRomHandle.type = DEVICE_TYPE_CART;
    sCartRomHandle.domain = PI_DOMAIN1;
    return &sCartRomHandle;
}

int osSetTimer(OSTimer* t, OSTime countdown, OSTime interval, OSMesgQueue* mq, OSMesg msg) {
    return 0;
}

int32_t osEPiStartDma(OSPiHandle* pihandle, OSIoMesg* mb, int32_t direction) {
    auto* decompMesg = reinterpret_cast<GdxDecompOSIoMesg*>(mb);
    if (decompMesg != nullptr) {
        decompMesg->piHandle = pihandle;

        if (direction == OS_READ && decompMesg->dramAddr != nullptr && decompMesg->size != 0) {
            /* Service cartridge reads from the loaded ROM image. devAddr is a
               physical cart address (0x10000000-based) or a plain ROM offset;
               masking covers both. Audio sample banks stream through here as
               the default AudioLoad DMA handler — zero-filling them produced
               fully silent synthesis output.

               The bytes are sourced through the single byte-source shim
               (GdxSegmentSourceRead) rather than copied straight out of
               gdx_rom_buffer, so audio and non-audio reads alike resolve in one
               place: the shim's containment table covers every blob family
               (segment_blob geometry plus the three audio_blob families) and any
               read outside a blob span falls through to the raw ROM inside the
               shim -- archive-first, with a byte-identical raw-ROM fallback. Its
               bounds check (romOffset+size > gdx_rom_size, or a null ROM image) is
               exactly the old guard and returns 0 in precisely the cases the old
               code zero-filled, so preserve that zero-fill here and out-of-range
               or absent reads still see deterministic contents. */
            const uint32_t romOffset = decompMesg->devAddr & 0x0FFFFFFFu;
            if (!GdxSegmentSourceRead(romOffset, decompMesg->size, decompMesg->dramAddr)) {
                memset(decompMesg->dramAddr, 0, decompMesg->size);
            }
        }

        if (decompMesg->hdr.retQueue != nullptr) {
            osSendMesg(decompMesg->hdr.retQueue, OS_MESG_PTR(mb), OS_MESG_NOBLOCK);
        }
    }

    return 0;
}

// Defined in port/gdx_audio_thread.cpp and queried below without a header include -- this file
// is compiled as part of the libultraship target, which has no include path onto port/. Same
// cross-module extern-declaration-without-header pattern already used elsewhere in this port
// (e.g. port/n64_sched.c's own forward declaration of gdx_audio_hle_run).
extern "C" int gdx_audio_thread_active(void);

uint32_t osAiGetLength() {
    // Real hardware returns the byte count still queued in the AI FIFO. We approximate it with
    // the host audio backend's queued-sample count (already-buffered interleaved s16 stereo
    // frames) converted to bytes (4 bytes per L/R sample pair), so AudioThread_CreateTaskImpl's
    // adaptive fill math (decomp/src/audio/disk/lib/thread.c, "samplesRemainingInAi =
    // osAiGetLength() / 4") throttles against real buffer occupancy instead of always assuming an
    // empty AI, which would make it over-produce forever.
    auto audio = Ship::Context::GetInstance() != nullptr ? Ship::Context::GetInstance()->GetAudio() : nullptr;
    std::shared_ptr<Ship::AudioPlayer> player = audio != nullptr ? audio->GetAudioPlayer() : nullptr;
    if (player == nullptr || !player->IsInitialized()) {
        return 0;
    }
    // Host jitter cushion: the game's adaptive fill (thread.c,
    // samplesRemainingInAi) targets the console AI FIFO depth — roughly one
    // 60Hz frame (~17ms) of audio. SDL pulls 1024-frame chunks and Windows
    // scheduling adds multi-ms jitter, so a 17ms cushion dips to empty
    // constantly; every dip plays as an audible silence hole (measured: 725
    // holes of ~10ms in a 113s capture). Under-report the queued amount so
    // the game's own fill math settles at target + cushion instead. 2048
    // frames = 64ms at 32kHz; must stay well under SDLAudioPlayer::DoPlay's
    // drop threshold. Tunable via GDX_AI_CUSHION (frames).
    //
    // The cushion only covers ordinary host scheduling jitter, and only for the legacy per-VI-tick
    // fiber producer, which has no catch-up mechanism of its own and cannot survive a real stall
    // (a long synchronous game-thread load blocks that same fiber outright; hitches up to ~131ms,
    // far longer than any cushion could cover). The dedicated audio thread
    // (port/gdx_audio_thread.cpp) instead runs `while (Buffered() < DesiredBuffered) produce()`
    // off the ACTUAL buffered amount, and is immune to game-thread stalls by construction (a real
    // OS thread, not a fiber sharing the stalled one), so under-reporting would only make it
    // over-produce. Report honestly while that thread is active, and restore the exact old cushion
    // when the kill switch (GDX_AUDIO_THREAD=0) reverts to the fiber path, leaving that path's
    // behavior unchanged.
    static int32_t sCushionFrames = -1;
    if (sCushionFrames < 0) {
        if (gdx_audio_thread_active()) {
            sCushionFrames = 0;
        } else {
            sCushionFrames = 2048;
            if (const char* env = std::getenv("GDX_AI_CUSHION")) {
                const long v = std::strtol(env, nullptr, 10);
                if (v >= 0 && v <= 4096) {
                    sCushionFrames = (int32_t)v;
                }
            }
        }
    }
    const int32_t buffered = (int32_t)player->Buffered() - sCushionFrames;
    return buffered > 0 ? (uint32_t)buffered * 4 : 0;
}

int32_t osAiSetNextBuffer(void* buff, size_t len) {
    // Forward the AI buffer submitted by AudioThread_CreateTaskImpl
    // (decomp/src/audio/disk/lib/thread.c, gAudioCtx.aiBuffers[index]) to the host audio backend.
    // This is the AI->host wiring only: it plays back whatever bytes are in that buffer. The PCM
    // itself comes from the port's aspMain ABI interpreter (port/n64_audio_hle.c), because the
    // decomp's audio pipeline only ever builds RSP command lists.
    if (buff == nullptr || len == 0) {
        return 0;
    }
    auto audio = Ship::Context::GetInstance() != nullptr ? Ship::Context::GetInstance()->GetAudio() : nullptr;
    std::shared_ptr<Ship::AudioPlayer> player = audio != nullptr ? audio->GetAudioPlayer() : nullptr;

    // Scan once; the underrun-resilience fallback below consumes this result.
    bool allZero = true;

    const uint8_t* samples = static_cast<const uint8_t*>(buff);
    for (size_t k = 0; k < len; k++) {
        if (samples[k] != 0) {
            allZero = false;
            break;
        }
    }

    // AI buffer underrun resilience: when the GAME thread runs a long
    // synchronous operation (course/segment asset loads, large mio0
    // decompresses) without yielding, the cooperative fiber scheduler cannot
    // run the AUDIO fiber for that whole stretch (see port/n64_sched.c), so
    // AudioSynth_Update never builds a command list for the missed tick(s) and
    // the buffer arriving here is all-zero -- measured as repeating few-ms
    // silence bursts during course loads. Emitting that silence verbatim is an
    // audible drop-out/click, so repeat the last buffer that actually had audio
    // in it, halving its gain on each consecutive miss so a longer stall decays
    // toward true silence instead of looping one snippet at full volume forever.
    // This softens the symptom only; the starvation itself is addressed by the
    // cooperative yields in Dma_LoadAssets (decomp/src/sys/dma.c) and mio0_decode
    // (torch/lib/libmio0/mio0.c).
    static std::vector<uint8_t> sLastGoodAiBuffer;
    static uint32_t sConsecutiveZeroAiBuffers = 0;
    std::vector<uint8_t> fadedSubstitute;
    const uint8_t* playBuf = static_cast<const uint8_t*>(buff);
    size_t playLen = len;

    // Gate the substitution off when the dedicated audio thread (port/gdx_audio_thread.cpp) is
    // driving playback. That thread owns its timing independently of the game/fiber scheduler and
    // so does not share the starvation mode above: an all-zero buffer arriving under it is
    // LEGITIMATE silence (no BGM/SFX between menu sounds), not an underrun. Substituting a decaying
    // copy of whatever last played turns that silence into audible ghost notes -- the intermittent
    // beeps/crackles seen with the thread on. Play the true (silent) buffer instead. The "last good
    // buffer" bookkeeping below is keyed on `!allZero`, not on this gate, so a gated-off all-zero
    // buffer neither resets the miss counter nor caches silence as "last good": if the kill switch
    // disables the thread at runtime, the fallback still has real audio to decay from.
    const bool substituteFade = allZero && !gdx_audio_thread_active();
    if (substituteFade) {
        if (!sLastGoodAiBuffer.empty() && sLastGoodAiBuffer.size() == len && (len % sizeof(int16_t)) == 0) {
            ++sConsecutiveZeroAiBuffers;
            // shift 1 => half gain, 2 => quarter, ... capped so it fully
            // decays to zero (int16 >> 16 is well-defined here, not UB, since
            // the shift amount is clamped below 16) rather than looping
            // forever at an audible volume during a very long stall.
            const uint32_t shift = sConsecutiveZeroAiBuffers > 15u ? 15u : sConsecutiveZeroAiBuffers;
            fadedSubstitute.resize(len);
            const int16_t* src = reinterpret_cast<const int16_t*>(sLastGoodAiBuffer.data());
            int16_t* dst = reinterpret_cast<int16_t*>(fadedSubstitute.data());
            const size_t numSamples = len / sizeof(int16_t);
            for (size_t s = 0; s < numSamples; s++) {
                dst[s] = static_cast<int16_t>(src[s] >> shift);
            }
            playBuf = fadedSubstitute.data();
            playLen = fadedSubstitute.size();
        }
    }
    if (!allZero) {
        sConsecutiveZeroAiBuffers = 0;
        if (sLastGoodAiBuffer.size() != len) {
            sLastGoodAiBuffer.resize(len);
        }
        std::memcpy(sLastGoodAiBuffer.data(), buff, len);
    }

    if (player == nullptr || !player->IsInitialized()) {
        return 0;
    }
    // Output reconstruction low-pass (matches the N64 audio DAC's output-stage rolloff
    // that accurate emulation models but a raw HLE pipeline omits -- the reference diff
    // measured +7.4dB of excess HF imaging near Nyquist that this removes). 4th-order
    // Butterworth (two cascaded biquads, RBJ cookbook) -- steep, no ringing, midrange
    // untouched. DEFAULT ON at 15kHz -- a gentle near-Nyquist rolloff chosen by listening
    // test: LLE audio removed the grain outright, so this only tames the top ~1kHz sliver
    // to match the console's analog reconstruction stage (11kHz was the HLE-era band-aid).
    // gdx-audio-lowpass.txt overrides the cutoff (a number in Hz) or disables it entirely
    // (contents "0" or "off"). Applied to a copy so the source buffer remains untouched.
    {
        static int sLastHz = -1;               /* -1 = not yet configured */
        static int sEnabled = 0;
        static double sB[2][3], sA[2][2];      /* [section]{b0,b1,b2}, {a1,a2} (a0 normalized) */
        static double sZ[2][2][2];             /* [channel][section]{x/y history} -> use DF2T state */
        static std::vector<int16_t> sBuf;
        /* Live cutoff from the ImGui Audio tab (F1 > Audio): gEnhancements.Audio.LowPassHz
         * (Hz; 0 = off; default 15000). Read each buffer on the audio thread (benign int race
         * with the menu); recompute the Butterworth coefficients ONLY when the value changes
         * (rare, on a menu edit). The filter history sZ is intentionally kept across a change so
         * moving the slider does not click. */
        int hz = CVarGetInteger("gEnhancements.Audio.LowPassHz", 15000);
        if (hz < 0) { hz = 0; }
        if (hz >= 16000) { hz = 15999; }
        if (hz != sLastHz) {
            sLastHz = hz;
            sEnabled = (hz > 0) ? 1 : 0;
            if (sEnabled) {
                const double kPi = 3.14159265358979323846;
                const double w0 = 2.0 * kPi * (double)hz / 32000.0;
                const double cw = std::cos(w0), sw = std::sin(w0);
                const double Q[2] = { 0.54119610, 1.30656296 }; /* 4th-order Butterworth section Qs */
                for (int s = 0; s < 2; s++) {
                    double alpha = sw / (2.0 * Q[s]);
                    double a0 = 1.0 + alpha;
                    sB[s][0] = ((1.0 - cw) / 2.0) / a0;
                    sB[s][1] = (1.0 - cw) / a0;
                    sB[s][2] = ((1.0 - cw) / 2.0) / a0;
                    sA[s][0] = (-2.0 * cw) / a0;
                    sA[s][1] = (1.0 - alpha) / a0;
                }
            }
        }
        if (sEnabled && (playLen % 4) == 0) {
            const size_t frames = playLen / 4;
            sBuf.resize(frames * 2);
            const int16_t* src = reinterpret_cast<const int16_t*>(playBuf);
            for (int ch = 0; ch < 2; ch++) {
                for (size_t i = 0; i < frames; i++) {
                    double x = (double)src[2 * i + ch];
                    for (int s = 0; s < 2; s++) { /* transposed direct form II */
                        double y = sB[s][0] * x + sZ[ch][s][0];
                        sZ[ch][s][0] = sB[s][1] * x - sA[s][0] * y + sZ[ch][s][1];
                        sZ[ch][s][1] = sB[s][2] * x - sA[s][1] * y;
                        x = y;
                    }
                    long v = (long)(x >= 0 ? x + 0.5 : x - 0.5);
                    if (v > 32767) v = 32767; else if (v < -32768) v = -32768;
                    sBuf[2 * i + ch] = (int16_t)v;
                }
            }
            playBuf = reinterpret_cast<const uint8_t*>(sBuf.data());
        }
    }
    // Master volume (FINAL output stage). CVar gEnhancements.Audio.MasterVolume (0..100, default
    // 100), registered by the port's GdxMenuBar ctor. Read live each buffer on the audio thread,
    // exactly like the low-pass above, so a menu edit applies without a restart (a benign int race
    // with the main-thread ImGui write: worst case one buffer sees the old value). Placed AFTER the
    // reconstruction low-pass so gain is the last thing done to the PCM before the device. At
    // vol==100 the multiply is skipped ENTIRELY, leaving playBuf untouched, so a default config is
    // sample-for-sample unmodified. Applied to a COPY (its own static scratch vector), never in
    // place on the source or the low-pass buffer, and guarded on (playLen % 4)==0 like the low-pass
    // so a ragged length is never misread as whole stereo s16 frames.
    {
        static std::vector<int16_t> sVolBuf;
        int vol = CVarGetInteger("gEnhancements.Audio.MasterVolume", 100);
        if (vol < 0) {
            vol = 0;
        }
        if (vol > 100) {
            vol = 100;
        }
        if (vol != 100 && (playLen % 4) == 0) {
            const size_t samples = playLen / sizeof(int16_t); /* interleaved stereo s16 */
            sVolBuf.resize(samples);
            const int16_t* src = reinterpret_cast<const int16_t*>(playBuf);
            const double gain = (double)vol / 100.0;
            for (size_t s = 0; s < samples; s++) {
                double scaled = (double)src[s] * gain;
                long v = (long)(scaled >= 0 ? scaled + 0.5 : scaled - 0.5); /* round half away from 0 */
                if (v > 32767) {
                    v = 32767;
                } else if (v < -32768) {
                    v = -32768;
                }
                sVolBuf[s] = (int16_t)v;
            }
            playBuf = reinterpret_cast<const uint8_t*>(sVolBuf.data());
        }
    }
    player->Play(playBuf, playLen);
    return 0;
}

int32_t __osMotorAccess(OSPfs* pfs, uint32_t vibrate) {
    auto io = Ship::Context::GetInstance()->GetControlDeck()->GetControllerByPort(pfs->channel)->GetRumble();
    if (vibrate) {
        io->StartRumble();
    } else {
        io->StopRumble();
    }

    return 0;
}

int32_t osMotorInit(OSMesgQueue* ctrlrqueue, OSPfs* pfs, int32_t channel) {
    pfs->channel = channel;
    return 0;
}
}
