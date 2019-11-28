include(${CMAKE_CURRENT_LIST_DIR}/common/include_guard.cmake)
cmake_include_guard()

# macros

include(${CMAKE_CURRENT_LIST_DIR}/global_macros.cmake)

# env

include(${CMAKE_CURRENT_LIST_DIR}/common/target_arch.cmake)
target_architecture(HOST_ARCH)

if(MSVC OR MSYS OR MINGW)
  set(HOST_OS Win)
elseif(APPLE)
  set(HOST_OS Mac)
elseif(UNIX)
  set(HOST_OS Linux)
else()
  message(FATAL_ERROR "Unsupported OS.")
endif()

status("")
status("Platform:")
status("  HOST_OS: ${HOST_OS}")
status("  HOST_ARCH: ${HOST_ARCH}")
status("  HOST_COMPILER: ${CMAKE_CXX_COMPILER_ID}")
status("    COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")

# options

include(${CMAKE_CURRENT_LIST_DIR}/global_options.cmake)

status("")
