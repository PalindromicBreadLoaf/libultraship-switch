#include "ship/audio/SDLAudioPlayer.h"
#include <spdlog/spdlog.h>

namespace Ship {

SDLAudioPlayer::~SDLAudioPlayer() {
    SPDLOG_TRACE("destruct SDL audio player");
    DoClose();
    SDL_QuitSubSystem(SDL_INIT_AUDIO);
}

void SDLAudioPlayer::DoClose() {
    if (mDevice != 0) {
        // Pause playback first
        SDL_PauseAudioDevice(mDevice, 1);
        // Clear any queued audio to prevent glitches when reopening
        SDL_ClearQueuedAudio(mDevice);
        SDL_CloseAudioDevice(mDevice);
        mDevice = 0;
    }
}

bool SDLAudioPlayer::DoInit() {
    if (SDL_Init(SDL_INIT_AUDIO) != 0) {
        SPDLOG_ERROR("SDL init error: {}", SDL_GetError());
        return false;
    }

    // Always open with the correct number of output channels
    mNumChannels = this->GetNumOutputChannels();

    SDL_AudioSpec want, have;
    SDL_zero(want);
    want.freq = this->GetSampleRate();
    want.format = AUDIO_S16SYS;
    want.channels = mNumChannels;
    want.samples = this->GetSampleLength();
    want.callback = NULL;

    mDevice = SDL_OpenAudioDevice(NULL, 0, &want, &have, 0);
    if (mDevice == 0) {
        SPDLOG_ERROR("SDL_OpenAudio error: {}", SDL_GetError());
        return false;
    }

    SPDLOG_INFO("SDL Audio initialized: {} channels, {} Hz", mNumChannels, this->GetSampleRate());

    SDL_PauseAudioDevice(mDevice, 0);
    return true;
}

int SDLAudioPlayer::Buffered() {
    return SDL_GetQueuedAudioSize(mDevice) / (sizeof(int16_t) * mNumChannels);
}

void SDLAudioPlayer::DoPlay(const uint8_t* buf, size_t len) {
    // Backpressure: drop this submission once the device queue is already well past the
    // configured target (GetDesiredBuffered() -- 4096 frames with the Phase 3 dedicated audio
    // thread active, port/gdx_audio_thread.cpp; the struct default of 2480 otherwise). Was
    // previously a hard, config-independent 6000-frame constant: harmless while
    // DesiredBuffered was 2480 (huge margin), but silently wrong once DesiredBuffered was
    // raised to 4096 (barely 1900 frames of margin left before an over-full queue starts
    // silently dropping submissions) or for any future lower-latency profile. The margin below
    // covers one extra production batch on top of the target so a producer slightly ahead of
    // schedule isn't punished (SAMPLES_HIGH in decomp/src/audio/disk/lib/thread.c's
    // audioBufferParameters is a few hundred frames per tick; 1024 comfortably covers both the
    // legacy fiber path's per-VI-tick batch and the dedicated thread's catch-up loop).
    static const int32_t kBacklogMargin = 1024;
    if (Buffered() < GetDesiredBuffered() + kBacklogMargin) {
        SDL_QueueAudio(mDevice, buf, len);
    }
}
} // namespace Ship
