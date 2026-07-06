#=================== Controller ===================

file(GLOB_RECURSE Source_Files__Controller RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "controller/*.cpp")

source_group("controller" FILES ${Source_Files__Controller})
target_sources(libultraship PRIVATE ${Source_Files__Controller})

#=================== Bridge ===================

source_group("bridge")

file(GLOB Source_Files__Bridge RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "bridge/*.cpp")

if(NOT ENABLE_SCRIPTING)
    list(FILTER Source_Files__Bridge EXCLUDE REGEX "bridge/scriptingbridge\\.cpp")
endif()

source_group("bridge" FILES ${Source_Files__Bridge})

target_sources(libultraship PRIVATE ${Source_Files__Bridge})

#=================== libultra ===================

source_group("libultra")

file(GLOB Source_Files__Libultra RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "libultra/*.cpp")

# G-Diffuser (F-Zero X port) R6: the port runs the decomp's OWN N64 cooperative scheduler
# (decomp/src/libultra/os/*.c). Disable libultraship's non-blocking message-queue stubs and
# VI stubs so the decomp's real blocking osRecvMesg/osSendMesg + our VI bridge are used instead.
list(FILTER Source_Files__Libultra EXCLUDE REGEX "libultra/os_mesg\\.cpp$")
list(FILTER Source_Files__Libultra EXCLUDE REGEX "libultra/os_vi\\.cpp$")

source_group("libultra" FILES ${Source_Files__Libultra})

target_sources(libultraship PRIVATE ${Source_Files__Libultra})

#=================== Log ===================

file(GLOB Source_Files__Log RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "log/*.cpp")

source_group("log" FILES ${Source_Files__Log})

target_sources(libultraship PRIVATE ${Source_Files__Log})

#=================== Window ===================

file(GLOB Source_Files__Window RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "window/*.cpp")

source_group("window" FILES ${Source_Files__Window})
target_sources(libultraship PRIVATE ${Source_Files__Window})

#=================== Gui ===================

file(GLOB_RECURSE Source_Files__Window__Gui RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "window/gui/*.cpp")

source_group("window/gui" FILES ${Source_Files__Window__Gui})
target_sources(libultraship PRIVATE ${Source_Files__Window__Gui})
