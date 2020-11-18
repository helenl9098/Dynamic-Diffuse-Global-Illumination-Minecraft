# RVPT
### Real-time Vulkan Path Tracer

A community project of the Graphics Programming Discord server.

This project sets out to create a runtime usable Path Tracer written using the Vulkan graphics API with minimal external dependencies. 

Features:
 * Compute shader based Path Tracing
 * Temporal Accumulation
 * Shader Hot-reloading
 * ImGui Integration
 * Rasterization Debug View

## :construction: WIP :construction: 

This project is still very much work-in-progress and is missing many features. Consider contributing if you want change it :)

## Build instructions

Requires:
 * C++17 compiler
 * CMake version 3.11 or higher
 * Vulkan SDK installed. Download here: https://vulkan.lunarg.com/sdk/home

CMake is the build system. 
Basic compilation instructions
```
git clone https://github.com/GraphicsProgramming/RVPT.git
cd RVPT
mkdir build
cd build
cmake .. 
cmake --build .
```
This will create the rvpt executable in the build directory. It requires the assets folder to operate currently.

Run the `rvpt` binary produced.

## TODO 
Things we would love to have but aren't quite there yet:
 * Model/Texture/Scene loading
 * Temporal Reprojection
 * PBR Material support
 * BVH acceleration
 * Skeltal animation
