# Building with ROCm PyTorch

Use ComfyUI's own Python throughout. Install a ROCm PyTorch build and matching
HIP SDK that support your GPU before installing these nodes. Do not replace
Torch with a CPU or NVIDIA build while installing dependencies.

On Windows, run native builds from an x64 Visual Studio Developer Command Prompt
with C++ Build Tools, the Windows SDK, CMake and Ninja available. Initialize the
ROCm SDK with the normal ComfyUI ROCm launcher. ROCM_HOME and HIP_PATH must point
to the physical matching SDK. The Windows HIP compiler is clang-cl.exe in its
lib/llvm/bin folder; set CXX to that compiler for GPU extension builds. WTiVo's
CPU CMake targets use the Microsoft C++ compiler instead.

From your ComfyUI directory, these Command Prompt settings select its Python:

```bat
set "COMFY_PYTHON=%CD%\python_env\python.exe"
set "MAX_JOBS=2"
set "DISTUTILS_USE_SDK=1"
set "MSSdk=1"
```

Use your actual Python path if the installation uses a different layout.
Run install_requirements.bat from the node directory, or use the commands below.
The batch file selects ComfyUI's Python and calls install.py. The Python
installer initializes the C++ tools and matching SDK, installs requirements,
builds the native backends and validates their imports. Run install.py --check
with ComfyUI's Python to check the runtime and compiler without installation.
Installation integrations may invoke install.py directly; the batch filename
alone does not cause automatic execution. __init__.py does not compile code. WTiVo also requires
VCPKG_ROOT and its CPU dependencies; VCPKG_INSTALLED_DIR may select an existing
manifest install tree. Bake Forger does not need native compilation. Native installers use PyTorch's HIP device properties to target discrete GPUs
by default, excluding integrated GPUs when a discrete GPU is visible. Multiple
discrete architectures are included once each. An explicit PYTORCH_ROCM_ARCH
setting takes precedence. Integrated-only systems target their integrated GPUs;
older PyTorch versions without GPU type information retain unclassified GPUs.
PyTorch's extension builder supplies the HIP compiler flags.
This does not make a compiled binary portable across every AMD architecture.

## WTiVo

Install numpy and trimesh using ComfyUI's Python. CPU builds require pybind11,
CGAL, Eigen, oneTBB and OpenVDB. The vcpkg manifest lists the CPU dependencies.
Set VCPKG_ROOT to your vcpkg checkout, install its x64-windows dependencies, and
set PYBIND11_DIR to pybind11's CMake directory in the same Python environment.

```bat
"%COMFY_PYTHON%" -m pip install -r requirements-runtime.txt -r requirements-build.txt
cmake -S . -B .build/cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl.exe -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows -DPython3_EXECUTABLE="%COMFY_PYTHON%" -Dpybind11_DIR="%PYBIND11_DIR%"
cmake --build .build/cpu --target wtivo_core wtivo_vdb
copy "%VCPKG_ROOT%\installed\x64-windows\bin\*.dll" build\
"%COMFY_PYTHON%" scripts/build_gpupr_hip.py
"%COMFY_PYTHON%" scripts/verify_hip_graph.py
```

Adjust the vcpkg installed directory if using a separate manifest install tree.
Keep the dependency DLLs beside the native modules in build/.

## Mesh Quad Reconstruct and CuMesh Decimate

Both nodes use the same bundled CuMesh HIP source. Build and install it once
with the Python environment running ComfyUI:

```bat
"%COMFY_PYTHON%" -m pip install -r requirements.txt
"%COMFY_PYTHON%" CuMesh-HIP/build_hip.py
"%COMFY_PYTHON%" -m pip install ./CuMesh-HIP --no-build-isolation --no-deps
```

The original install.py and Windows launchers are retained with AMD changes.
Mesh Quad's installer accepts its original --build-if-needed option.
CuMesh Decimate's installer builds the bundled backend if none is installed.

## Trellis2 Mesh Encoder

```bat
"%COMFY_PYTHON%" -m pip install -r requirements.txt
"%COMFY_PYTHON%" HIP-runtime/build_hip.py
"%COMFY_PYTHON%" HIP-runtime/install_runtime.py
```

The runtime installer backs up existing trellis2 and o_voxel folders before
replacing them, checks imports, and restores the previous folders if validation fails. This port uses ComfyUI's
Trellis2 sparse convolution and VAE support. Obtain encoder JSON, weights and
the matching shape VAE as described in the original node documentation.

## LODTailor Bake Forger

Use Blender with Cycles HIP support. This node does not compile a Torch GPU
extension. Its original device settings and workflow are retained.

## Previously tested environment

Windows 11, RX 9070 XT (gfx1201), Python 3.12.9,
Torch 2.15.0a0+rocm10.2.0a20260926, HIP 7.17.26384 and ROCm SDK 10.2.
Blender 5.2 was used for Bake Forger. This records validation, not a universal
version requirement for building source. Other GPU/runtime combinations need
their own validation. Close ComfyUI before replacing native modules.
