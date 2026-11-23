cmake_minimum_required(VERSION 3.20)

# Must precede project(). macOS 12 is older than any Resolve that loads Metal
# OFX plugins, so it costs nothing and keeps the bundle loadable on older
# machines; the newer Metal API the plugin uses is behind @available.
set(CMAKE_OSX_DEPLOYMENT_TARGET "12.0" CACHE STRING "Minimum macOS version")

project(ResolveShatterPhysics VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(OFX_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/external/openfx)
if(NOT EXISTS ${OFX_ROOT}/Support/Library/ofxsCore.cpp)
  message(FATAL_ERROR "OpenFX SDK missing. Run: git clone --depth 1 "
                      "https://github.com/AcademySoftwareFoundation/openfx.git external/openfx")
endif()

# --------------------------------------------------------------- core library
# Host-independent: no OFX headers anywhere under src/core, which is what lets
# the offline harness and the tests link the identical code Resolve runs.
add_library(shp_core STATIC
  src/core/ShardGeom.cpp
  src/core/Segmentation.cpp
  src/core/Cells.cpp
  src/core/Compositor.cpp
  src/core/EdgeField.cpp
  src/core/Physics2D.cpp
  src/core/SimCache.cpp
)
target_include_directories(shp_core PUBLIC src src/core)
if(MSVC)
  target_compile_options(shp_core PRIVATE /W4 /fp:fast)
else()
  # Deliberately no -ffast-math to pair with MSVC's /fp:fast. The determinism
  # the simulation needs is run-to-run on ONE machine, which plain IEEE gives
  # for free, and fast-math is entitled to assume no NaNs -- which would quietly
  # delete the guards that keep one bad centroid from poisoning a cached run.
  target_compile_options(shp_core PRIVATE -Wall -Wextra)
endif()

# ------------------------------------------------------------------- OFX plugin
# The Support library is compiled straight into the plugin. Consuming the openfx
# repo's own CMake drags in OpenGL/CUDA options and example plugins.
add_library(ShatterPhysics SHARED
  src/ShatterPlugin.cpp
  src/interact/Overlay.cpp
  ${OFX_ROOT}/Support/Library/ofxsCore.cpp
  ${OFX_ROOT}/Support/Library/ofxsImageEffect.cpp
  ${OFX_ROOT}/Support/Library/ofxsInteract.cpp
  ${OFX_ROOT}/Support/Library/ofxsLog.cpp
  ${OFX_ROOT}/Support/Library/ofxsMultiThread.cpp
  ${OFX_ROOT}/Support/Library/ofxsParams.cpp
  ${OFX_ROOT}/Support/Library/ofxsProperty.cpp
  ${OFX_ROOT}/Support/Library/ofxsPropertyValidation.cpp
)
target_link_libraries(ShatterPhysics PRIVATE shp_core)
target_include_directories(ShatterPhysics PRIVATE
  ${OFX_ROOT}/include
  ${OFX_ROOT}/Support/include
  ${OFX_ROOT}/Support/Plugins/include
)
set_target_properties(ShatterPhysics PROPERTIES PREFIX "" SUFFIX ".ofx")

if(NOT MSVC)
  # Export only the OFX entry points (marked OfxExport), which is what a Windows
  # DLL does by default. Otherwise every OFX Support and core symbol is public,
  # and a host that loads several plugins built on the same Support library
  # sees hundreds of identically named C++ symbols from each of them.
  set_target_properties(ShatterPhysics shp_core PROPERTIES
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
endif()

if(MSVC)
  # Static CRT: an OFX bundle is loaded by a host that will not have our
  # redistributables, and a missing VCRUNTIME dll fails as a silent no-load --
  # the plugin simply never appears in Resolve's list, with no error anywhere.
  set_property(TARGET ShatterPhysics shp_core PROPERTY
               MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  target_compile_definitions(ShatterPhysics PRIVATE
                             _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# ------------------------------------------------------------------- CUDA path
# Optional: without a CUDA toolkit the plugin still builds and runs on the CPU.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  # Must be set BEFORE enable_language: that call installs a default, after which
  # an "if(NOT DEFINED ...)" guard silently never fires and the plugin ships
  # PTX-only code that the driver has to JIT.
  # CUDA 13 dropped everything below Turing. 120 is Blackwell (RTX 50-series).
  set(CMAKE_CUDA_ARCHITECTURES 75 86 89 120 CACHE STRING "CUDA archs to build" FORCE)
  enable_language(CUDA)

  # Built as its own static library so the plugin and the parity test link the
  # SAME compiled kernel, rather than two builds of "the same" code -- which is
  # precisely the situation a parity test exists to rule out.
  add_library(shp_cuda STATIC src/render/CudaKernel.cu)
  target_include_directories(shp_cuda PUBLIC src src/core)
  target_compile_definitions(shp_cuda PUBLIC SHP_WITH_CUDA=1)
  set_target_properties(shp_cuda PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION OFF
    # Static runtime for the same reason as the CRT: a missing cudart DLL shows
    # up as the plugin silently not appearing in Resolve.
    CUDA_RUNTIME_LIBRARY Static
    POSITION_INDEPENDENT_CODE ON
  )
  if(MSVC)
    set_property(TARGET shp_cuda PROPERTY
                 MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  endif()

  target_link_libraries(ShatterPhysics PRIVATE shp_cuda)
  target_compile_definitions(ShatterPhysics PRIVATE SHP_WITH_CUDA=1)
  set_target_properties(ShatterPhysics PROPERTIES CUDA_RUNTIME_LIBRARY Static)
  message(STATUS "CUDA enabled (archs: ${CMAKE_CUDA_ARCHITECTURES})")
else()
  target_compile_definitions(ShatterPhysics PRIVATE SHP_WITH_CUDA=0)
  message(STATUS "CUDA toolkit not found - building CPU-only")
endif()

# ------------------------------------------------------------------ Metal path
# macOS only. Unlike CUDA there is no toolkit to look for: Metal ships with the
# OS, so every Mac build has it, and the plugin decides at load time whether to
# claim it (MetalAvailable() needs a device AND a kernel that compiled).
#
# The kernel is NOT compiled here. The offline `metal` compiler comes only with
# full Xcode, not the Command Line Tools, so MetalKernel.metal is embedded into
# the binary as a string and compiled by the Metal framework on first use.
if(APPLE)
  enable_language(OBJCXX)

  # Embedded at BUILD time, so editing the kernel always recompiles the host
  # code that carries it -- see cmake/EmbedText.cmake for why not at configure.
  set(SHP_METAL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/render/MetalKernel.metal)
  set(SHP_METAL_HDR ${CMAKE_BINARY_DIR}/generated/MetalKernelSource.h)
  add_custom_command(
    OUTPUT ${SHP_METAL_HDR}
    COMMAND ${CMAKE_COMMAND} -DIN=${SHP_METAL_SRC} -DOUT=${SHP_METAL_HDR}
            -DSYMBOL=kMetalKernelSource -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedText.cmake
    DEPENDS ${SHP_METAL_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedText.cmake
    COMMENT "Embedding MetalKernel.metal"
    VERBATIM)

  # Its own static library for the same reason as shp_cuda: the plugin and the
  # parity test link the SAME compiled host code and the same embedded kernel.
  add_library(shp_metal STATIC src/render/MetalRender.mm ${SHP_METAL_HDR})
  target_include_directories(shp_metal PUBLIC src src/core
                             PRIVATE ${CMAKE_BINARY_DIR}/generated)
  target_compile_options(shp_metal PRIVATE -fobjc-arc -Wall -Wextra)
  target_compile_definitions(shp_metal PUBLIC SHP_WITH_METAL=1)
  target_link_libraries(shp_metal PUBLIC "-framework Metal" "-framework Foundation")
  set_target_properties(shp_metal PROPERTIES
    CXX_VISIBILITY_PRESET hidden OBJCXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)

  target_link_libraries(ShatterPhysics PRIVATE shp_metal)
  message(STATUS "Metal enabled")
endif()

# Assemble the .ofx.bundle next to the binary.
#
# The architecture directory is part of the OFX spec and the host will not look
# anywhere else, so it is not a cosmetic path: a macOS build dropped into Win64
# simply never loads, with no error to explain why.
#
# Apple Silicon has no directory of its own in the spec -- a Mach-O binary
# carries its own architecture, so arm64 and universal builds both live in
# Contents/MacOS.
if(APPLE)
  set(OFX_ARCH_DIR MacOS)
elseif(WIN32)
  set(OFX_ARCH_DIR Win64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(OFX_ARCH_DIR Linux-arm-64)
else()
  set(OFX_ARCH_DIR Linux-x86-64)
endif()

set(BUNDLE_ROOT ${CMAKE_BINARY_DIR}/bundle/ShatterPhysics.ofx.bundle/Contents)
set(BUNDLE ${BUNDLE_ROOT}/${OFX_ARCH_DIR})
add_custom_command(TARGET ShatterPhysics POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BUNDLE}
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:ShatterPhysics> ${BUNDLE}/ShatterPhysics.ofx
  # Required on macOS, where a bundle without it is not a bundle. Harmless
  # elsewhere, and the siblings ship it on every platform for that reason.
  COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_CURRENT_SOURCE_DIR}/packaging/Info.plist ${BUNDLE_ROOT}/Info.plist
  COMMENT "Assembling ShatterPhysics.ofx.bundle (${OFX_ARCH_DIR})"
)

# ---------------------------------------------------------------- test harness
# Offline: PNG in, shattered frames out, linking the same core the plugin does.
# Tuning a shatter through this costs a second; through Resolve it costs a
# relaunch.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/shattertest.cpp)
  add_executable(shattertest tools/shattertest.cpp)
  target_link_libraries(shattertest PRIVATE shp_core)
  target_include_directories(shattertest PRIVATE external/stb)
  if(MSVC)
    set_property(TARGET shattertest PROPERTY
                 MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    target_compile_definitions(shattertest PRIVATE _CRT_SECURE_NO_WARNINGS)
  endif()
endif()

# ----------------------------------------------------------------------- tests
# One executable rather than one per property: these all share the same
# synthetic test frame, and building it once keeps the suite under a second.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/shattertests.cpp)
  enable_testing()
  add_executable(shattertests tests/shattertests.cpp)
  target_link_libraries(shattertests PRIVATE shp_core)
  if(MSVC)
    set_property(TARGET shattertests PROPERTY
                 MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    target_compile_definitions(shattertests PRIVATE _CRT_SECURE_NO_WARNINGS)
  endif()
  add_test(NAME shatter_properties COMMAND shattertests)

  # CPU-vs-GPU parity. Skipped at runtime when no device is present, so the
  # suite still passes on a machine without a GPU.
  if(CMAKE_CUDA_COMPILER AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/paritytest.cpp)
    add_executable(paritytest tests/paritytest.cpp)
    target_link_libraries(paritytest PRIVATE shp_core shp_cuda)
    set_target_properties(paritytest PROPERTIES CUDA_RUNTIME_LIBRARY Static)
    if(MSVC)
      set_property(TARGET paritytest PROPERTY
                   MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
      target_compile_definitions(paritytest PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
    add_test(NAME cpu_gpu_parity COMMAND paritytest)
  endif()

  # CPU-vs-Metal parity. Unlike the CUDA one this is the only thing keeping the
  # two in agreement, because the Metal kernel is a port rather than the shared
  # ShadePixel -- see src/render/MetalRender.h.
  if(APPLE AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/metalparity.cpp)
    add_executable(metalparity tests/metalparity.cpp)
    target_link_libraries(metalparity PRIVATE shp_core shp_metal)
    add_test(NAME cpu_metal_parity COMMAND metalparity)
  endif()
endif()
