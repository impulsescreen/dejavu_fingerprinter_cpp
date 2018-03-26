cmake_minimum_required(VERSION 3.2)
project(libismotg-music)

set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wno-deprecated-declarations")
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -D__FILENAME__='\"$(subst ${CMAKE_SOURCE_DIR}/,,$(abspath $<))\"'")
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS} -g3 -O0 -D_DEBUG -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")

include_directories ("${PROJECT_SOURCE_DIR}/kiss_fft130" )
include_directories ("${PROJECT_SOURCE_DIR}/kiss_fft130/tools" )

set (FFT_LIBRARIES
    kiss_fft
)

set (SOURCE_FILES
    DejavuFingerprinter
)

add_subdirectory(kiss_fft130)

add_library(ismotg-music STATIC ${SOURCE_FILES})

