#=================== ImGui ===================
target_sources(ImGui
	PRIVATE
	${imgui_SOURCE_DIR}/backends/imgui_impl_dx11.cpp
	${imgui_SOURCE_DIR}/backends/imgui_impl_win32.cpp
)

# The Win32 backend polls gamepads through XInput only and clears
# ImGuiBackendFlags_HasGamepad every frame when no XInput device answers,
# which discards externally fed ImGuiKey_Gamepad* nav events (non-XInput pads
# such as a raw DualSense). Gamepad nav is fed from SDL by the host app
# instead, so disable the backend's XInput path entirely.
target_compile_definitions(ImGui PRIVATE IMGUI_IMPL_WIN32_DISABLE_GAMEPAD)

find_package(SDL2 CONFIG REQUIRED)
target_link_libraries(ImGui PUBLIC SDL2::SDL2 SDL2::SDL2main)

find_package(GLEW REQUIRED)
target_link_libraries(ImGui PUBLIC opengl32 GLEW::GLEW)
