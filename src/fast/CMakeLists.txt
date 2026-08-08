#=================== Top-Level ===================
file(GLOB Source_Files__TopLevel RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")

source_group("" FILES ${Source_Files__TopLevel})
target_sources(libultraship PRIVATE ${Source_Files__TopLevel})

#=================== Debug ===================

file(GLOB Source_Files__Debug RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "debug/*.cpp")

source_group("debug" FILES ${Source_Files__Debug})
target_sources(libultraship PRIVATE ${Source_Files__Debug})

#=================== Resource ===================

file(GLOB Source_Files__Resource__Types RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "resource/type/*.h" "resource/type/*.cpp")
source_group("resource/type" FILES ${Source_Files__Resource__Types})

file(GLOB Source_Files__Resource__Factories RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "resource/factory/*.h" "resource/factory/*.cpp")
source_group("resource/factory" FILES ${Source_Files__Resource__Factories})

target_sources(libultraship PRIVATE ${Source_Files__Resource} ${Source_Files__Resource__Types} ${Source_Files__Resource__Factories} ${Source_Files__Resource__Archive})

#=================== Graphic ===================

file(GLOB Source_Files__Graphic RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "backends/*.cpp")

if (NOT CMAKE_SYSTEM_NAME STREQUAL "Windows")
    list(FILTER Source_Files__Graphic EXCLUDE REGEX "graphic/Fast3D/backends/gfx_dxgi*")
    list(FILTER Source_Files__Graphic EXCLUDE REGEX "graphic/Fast3D/backends/gfx_direct3d*")
    list(FILTER Source_Files__Graphic EXCLUDE REGEX "graphic/Fast3D/backends/dxsdk/*")
endif()

if (NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    list(FILTER Source_Files__Graphic EXCLUDE REGEX "graphic/Fast3D/backends/gfx_metal*")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
    list(FILTER Source_Files__Graphic EXCLUDE REGEX "graphic/Fast3D/backends/gfx_opengl*")
endif()

source_group("graphic" FILES ${Source_Files__Graphic})
target_sources(libultraship PRIVATE ${Source_Files__Graphic})

#=================== Shader cache fingerprint ===================
#
# gfx_shader_cache.cpp keeps compiled shader payloads across runs. A stored blob is only valid
# for the source its generator emits today, so editing an emitter or a template would otherwise
# resurrect wrong bytecode on a user's machine with nothing pointing back here. Hash the
# generator inputs at configure time; the cache discards any store whose fingerprint differs.
#
# The CMAKE_CONFIGURE_DEPENDS below is load-bearing: without it a template edit never re-runs
# configure and the compiled-in digest goes on describing a build that no longer exists.
set(GDX_SHADER_FINGERPRINT_INPUTS
    "${CMAKE_CURRENT_SOURCE_DIR}/backends/gfx_direct3d_common.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/backends/gfx_direct3d11.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/backends/gfx_opengl.cpp"
)
file(GLOB_RECURSE GDX_SHADER_FINGERPRINT_TEMPLATES "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*")
list(APPEND GDX_SHADER_FINGERPRINT_INPUTS ${GDX_SHADER_FINGERPRINT_TEMPLATES})
list(SORT GDX_SHADER_FINGERPRINT_INPUTS)

set(GDX_SHADER_FINGERPRINT_ACC "")
foreach(gdxShaderInput IN LISTS GDX_SHADER_FINGERPRINT_INPUTS)
    if(EXISTS "${gdxShaderInput}" AND NOT IS_DIRECTORY "${gdxShaderInput}")
        file(SHA256 "${gdxShaderInput}" gdxShaderInputHash)
        string(APPEND GDX_SHADER_FINGERPRINT_ACC "${gdxShaderInputHash}")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${gdxShaderInput}")
    endif()
endforeach()

# 64 bits is deliberate: a change detector, not a security boundary, and a u64 is what the store
# header carries. An all-zero digest reads as "no fingerprint" and merely disables the cache.
string(SHA256 GDX_SHADER_FINGERPRINT_FULL "${GDX_SHADER_FINGERPRINT_ACC}")
string(SUBSTRING "${GDX_SHADER_FINGERPRINT_FULL}" 0 16 GDX_SHADER_FINGERPRINT_HEX)

target_compile_definitions(libultraship PRIVATE GDX_SHADER_CACHE_FINGERPRINT=0x${GDX_SHADER_FINGERPRINT_HEX}ull)
message(STATUS "G-Diffuser shader cache fingerprint: 0x${GDX_SHADER_FINGERPRINT_HEX}")
