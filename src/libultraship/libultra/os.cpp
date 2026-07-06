#include "libultraship/libultraship.h"
#include <SDL2/SDL.h>
#include <cstdio>
#include <cstring>
#include <ratio>

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
            memset(decompMesg->dramAddr, 0, decompMesg->size);
        }

        if (decompMesg->hdr.retQueue != nullptr) {
            osSendMesg(decompMesg->hdr.retQueue, OS_MESG_PTR(mb), OS_MESG_NOBLOCK);
        }
    }

    return 0;
}

uint32_t osAiGetLength() {
    // R7 (audio slice): real hardware returns the byte count still queued in the AI FIFO.
    // We approximate it with the host audio backend's queued-sample count (already-buffered
    // interleaved s16 stereo frames) converted to bytes (4 bytes per L/R sample pair), so
    // AudioThread_CreateTaskImpl's adaptive fill math (decomp/src/audio/disk/lib/thread.c:52,
    // "samplesRemainingInAi = osAiGetLength() / 4") throttles against real buffer occupancy
    // instead of always assuming an empty AI (which would make it always over-produce).
    // This does not by itself make sound audible — see osAiSetNextBuffer below.
    auto audio = Ship::Context::GetInstance() != nullptr ? Ship::Context::GetInstance()->GetAudio() : nullptr;
    std::shared_ptr<Ship::AudioPlayer> player = audio != nullptr ? audio->GetAudioPlayer() : nullptr;
    if (player == nullptr || !player->IsInitialized()) {
        return 0;
    }
    return (uint32_t)player->Buffered() * 4;
}

int32_t osAiSetNextBuffer(void* buff, size_t len) {
    // R7 (audio slice): forward the AI buffer submitted by AudioThread_CreateTaskImpl
    // (decomp/src/audio/disk/lib/thread.c:56, gAudioCtx.aiBuffers[index]) to the host audio
    // backend. This is the AI->host wiring only: on its own it plays back whatever bytes are
    // in that buffer. Producing real PCM there still requires an aspMain ABI interpreter (the
    // decomp's audio pipeline only ever builds RSP command lists — see the audio slice
    // engram record 'slice/audio' for exact scope); until that exists, this call is safe but
    // will play silence/stale buffer contents rather than music/SFX.
    if (buff == nullptr || len == 0) {
        return 0;
    }
    auto audio = Ship::Context::GetInstance() != nullptr ? Ship::Context::GetInstance()->GetAudio() : nullptr;
    std::shared_ptr<Ship::AudioPlayer> player = audio != nullptr ? audio->GetAudioPlayer() : nullptr;

    // TEMP one-shot diagnostics (audio-silence triage): distinguishes the three
    // failure sites in one run — no [ai] lines = submission never happens;
    // zero=1 = interpreter produced silence (input side); zero=0 but no sound =
    // SDL output side. Appends to the same log the port writes. Remove once
    // audio is confirmed audible.
    {
        static int sAiDiag = 0;
        if (sAiDiag < 6) {
            ++sAiDiag;
            bool allZero = true;
            const uint8_t* p = static_cast<const uint8_t*>(buff);
            for (size_t k = 0; k < len; k++) {
                if (p[k] != 0) {
                    allZero = false;
                    break;
                }
            }
            FILE* f = fopen("gdiffuser-run.log", "ab");
            if (f != nullptr) {
                fprintf(f, "[ai] submit #%d len=%zu zero=%d player=%d init=%d\n", sAiDiag, len, allZero ? 1 : 0,
                        player != nullptr ? 1 : 0, (player != nullptr && player->IsInitialized()) ? 1 : 0);
                fclose(f);
            }
        }
    }

    if (player == nullptr || !player->IsInitialized()) {
        return 0;
    }
    player->Play(static_cast<const uint8_t*>(buff), len);
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
