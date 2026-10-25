cmake_minimum_required(VERSION 3.20)
project(MetalGaussianBlur LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dynamically finds all .cpp, .mm, and .metal files in your directory
file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp" "*.mm" "*.metal" "*.h" "*.hpp")

add_executable(MetalGaussianBlur ${SOURCES})

find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)
find_library(COCOA_FRAMEWORK Cocoa REQUIRED)

target_link_libraries(MetalGaussianBlur PRIVATE 
    ${METAL_FRAMEWORK} 
    ${METALKIT_FRAMEWORK} 
    ${COCOA_FRAMEWORK}
)

set_target_properties(MetalGaussianBlur PROPERTIES
    MACOSX_BUNDLE TRUE
    MACOSX_BUNDLE_BUNDLE_NAME "Metal Gaussian Blur"
    MACOSX_BUNDLE_GUI_IDENTIFIER "com.sebastian.metalgaussianblur"
    MACOSX_BUNDLE_BUNDLE_VERSION "1.0.0"
    MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0"
)
