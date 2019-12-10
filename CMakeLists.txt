cmake_minimum_required(VERSION 3.0)
project(start-opencv VERSION 1.0.0 LANGUAGES C CXX)

include(cmake/global.cmake)

set(MY_ROOT ${PROJECT_SOURCE_DIR})
set(MY_DATA ${MY_ROOT}/data)

# cmake options

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
endif()
if(NOT CMAKE_DEBUG_POSTFIX)
  set(CMAKE_DEBUG_POSTFIX d)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
#set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)

# flags

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wall -Wformat")
if(NOT HOST_OS STREQUAL "Win")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
endif()
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

string(STRIP "${CMAKE_CXX_FLAGS}" CMAKE_CXX_FLAGS)
status("CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

# packages

LIST(APPEND CMAKE_MODULE_PATH "${MY_ROOT}/cmake/modules")

## opencv

find_package(OpenCV 4 REQUIRED)
status_var(OpenCV_INCLUDE_DIRS)
status_var(OpenCV_LIBS)

# targets

set(MY_OUTPUT "${MY_ROOT}/_output")
set_outdir(
  ARCHIVE "${MY_OUTPUT}/lib"
  LIBRARY "${MY_OUTPUT}/lib"
  RUNTIME "${MY_OUTPUT}/bin"
)

include_directories(
  ${MY_ROOT}/src/cpp
  ${OpenCV_INCLUDE_DIRS}
)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

add_subdirectory(src/cpp)
