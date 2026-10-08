from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.build_cache import file_identity, tree_identity, cache_directory


NODE_DIR = Path(__file__).resolve().parent


def _run(command, env=None):
    command = [str(item) for item in command]
    if env is not None:
        executable = shutil.which(command[0], path=env.get("PATH"))
        if executable:
            command[0] = executable
    print("[Installer]", subprocess.list2cmdline(command), flush=True)
    subprocess.check_call(command, cwd=NODE_DIR, env=env)


def _gpu_build_architectures(env, torch):
    # Respect an explicit target supplied by the user or their launcher.
    if env.get("PYTORCH_ROCM_ARCH", "").strip():
        print("[Installer] GPU architectures (override):", env["PYTORCH_ROCM_ARCH"])
        return
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        arch = getattr(props, "gcnArchName", "").split(":", 1)[0]
        if not arch:
            raise RuntimeError("Cannot determine the HIP architecture for " + props.name +
                               ". Set PYTORCH_ROCM_ARCH explicitly.")
        integrated = getattr(props, "is_integrated", getattr(props, "integrated", None))
        devices.append((props.name, arch, integrated))
    if not devices:
        raise RuntimeError("No visible ROCm GPU is available for compilation.")
    # Older PyTorch builds may lack the HIP integrated-device property.
    # Keep unclassified devices rather than guessing from names or VRAM.
    selected = [device for device in devices if device[2] != 1]
    if not selected:
        selected = devices
        print("[Installer] Only integrated GPUs are visible; targeting those GPUs.")
    for name, arch, integrated in devices:
        if integrated is None:
            print("[Installer] GPU type unavailable; retaining:", name, arch)
        elif (name, arch, integrated) not in selected:
            print("[Installer] Excluding integrated GPU:", name, arch)
    env["PYTORCH_ROCM_ARCH"] = ";".join(dict.fromkeys(device[1] for device in selected))
    print("[Installer] GPU build architectures:", env["PYTORCH_ROCM_ARCH"])


def _default_jobs():
    """Parallel compile jobs: half the logical CPUs, at most one per 2.5 GB of free RAM, 2-16."""
    try:
        import psutil
        ram_jobs = int(psutil.virtual_memory().available / (2.5 * 1024**3))
    except Exception:
        return 2
    return max(2, min((os.cpu_count() or 4) // 2, ram_jobs, 16))


def _native_environment():
    if os.name != "nt":
        raise RuntimeError("This native build currently supports Windows x64.")
    include = Path(sysconfig.get_paths()['include']) / 'Python.h'
    library = Path(sys.base_prefix) / 'libs' / ('python'+str(sys.version_info.major)+str(sys.version_info.minor)+'.lib')
    if not include.is_file() or not library.is_file():
        raise RuntimeError('ComfyUI Python development files are missing: '+str(include)+' or '+str(library)+'. Native compilation needs matching Python headers and the import library.')
    env = {key.upper(): value for key, value in os.environ.items()}
    sdk = env.get("ROCM_HOME") or env.get("HIP_PATH") or env.get("ROCM_PATH")
    if not sdk:
        for name in ("_rocm_sdk_core", "_rocm_sdk_devel"):
            spec = importlib.util.find_spec(name)
            if spec and spec.origin:
                candidate = Path(spec.origin).parent
                if (candidate / "lib/llvm/bin/clang-cl.exe").is_file():
                    sdk = str(candidate)
                    break
    if not sdk or not (Path(sdk) / "lib/llvm/bin/clang-cl.exe").is_file():
        raise RuntimeError("Set ROCM_HOME to the matching Windows HIP SDK directory.")

    vswhere = Path(env.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    if not vswhere.is_file():
        raise RuntimeError("Install Visual Studio C++ Build Tools and the Windows SDK.")
    installation = subprocess.check_output([
        str(vswhere), "-latest", "-products", "*", "-requires",
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath",
    ], text=True).strip()
    if not installation:
        raise RuntimeError("Visual Studio x64 C++ Build Tools were not found.")
    vs = Path(installation)
    script = NODE_DIR / ".build" / "developer-environment.cmd"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text('@echo off\ncall "' + str(vs / "Common7/Tools/VsDevCmd.bat") + '" -no_logo -arch=x64 -host_arch=x64 >nul\nif errorlevel 1 exit /b 1\nset\n', encoding="utf-8")
    vcpkg_settings = {key: env[key] for key in ("VCPKG_ROOT", "VCPKG_INSTALLED_DIR") if key in env}
    output = subprocess.check_output(["cmd.exe", "/d", "/c", str(script)], text=True, env=env)
    for line in output.splitlines():
        if "=" in line and not line.startswith("="):
            key, value = line.split("=", 1)
            env[key.upper()] = value
    env.update(vcpkg_settings)
    cmake_tools = vs / "Common7/IDE/CommonExtensions/Microsoft/CMake"
    env["PATH"] = os.pathsep.join([
        str(cmake_tools / "Ninja"), str(cmake_tools / "CMake/bin"),
        str(Path(sys.executable).parent / "Scripts"), str(Path(sdk) / "bin"), env.get("PATH", ""),
    ])
    env["ROCM_HOME"] = sdk
    env["HIP_PATH"] = sdk
    env["CXX"] = str(Path(sdk) / "lib/llvm/bin/clang-cl.exe")
    env["DISTUTILS_USE_SDK"] = "1"
    env["MSSDK"] = "1"
    env.setdefault("MAX_JOBS", str(_default_jobs()))
    for tool in ("cl.exe", "ninja.exe", "cmake.exe"):
        if not shutil.which(tool, path=env["PATH"]):
            raise RuntimeError("Required build tool is missing: " + tool)
    return env


def _install_cpu_dependencies(env):
    vcpkg = env.get("VCPKG_ROOT")
    root = Path(vcpkg) if vcpkg else NODE_DIR / '.deps/vcpkg'
    # Visual Studio's bundled tool may omit the ports registry. Use a complete
    # node-local checkout in that case rather than writing under Program Files.
    if not (root/'ports').is_dir() or not (root/'scripts/buildsystems/vcpkg.cmake').is_file():
        root = NODE_DIR / '.deps/vcpkg'
        if not root.exists():
            root.parent.mkdir(parents=True,exist_ok=True)
            _run(['git','clone','--depth','1','https://github.com/microsoft/vcpkg.git',root],env)
        if not (root/'ports').is_dir() or not (root/'bootstrap-vcpkg.bat').is_file():
            raise RuntimeError('Incomplete vcpkg checkout at '+str(root)+'. Preserve it and repair the checkout before retrying.')
    executable = root/'vcpkg.exe'
    if not executable.is_file():
        # Run from the checkout so cmd receives no quoted absolute batch path.
        # Python's list quoting would otherwise turn embedded quotes into \".
        command = [shutil.which('cmd.exe',path=env.get('PATH')) or 'cmd.exe',
                   '/d','/c','call bootstrap-vcpkg.bat -disableMetrics']
        print('[Installer]',subprocess.list2cmdline(command),flush=True)
        subprocess.check_call(command,cwd=root,env=env)
    installed = NODE_DIR / '.deps/vcpkg_installed'
    work = NODE_DIR / '.deps/vcpkg-work'
    work.mkdir(parents=True,exist_ok=True)
    env['VCPKG_ROOT'] = str(root)
    env['VCPKG_INSTALLED_DIR'] = str(installed)
    print('[Installer] Installing Eigen3, CGAL, OpenVDB and TBB from vcpkg.json.',flush=True)
    _run([executable,'install','--triplet=x64-windows','--host-triplet=x64-windows',
          '--x-manifest-root='+str(NODE_DIR),'--x-install-root='+str(installed),
          '--x-buildtrees-root='+str(work/'buildtrees'),
          '--x-packages-root='+str(work/'packages'),'--downloads-root='+str(work/'downloads')],env)
    return root,installed


def _snapshot_build():
    """Back up the installed modules so a failed install can be undone."""
    build = NODE_DIR / "build"
    backup = NODE_DIR / ".build" / "installed-backup"
    shutil.rmtree(backup, ignore_errors=True)
    if build.is_dir():
        # Windows keeps a loaded .pyd locked; replacing it would fail half-way.
        for module in build.glob("*.pyd"):
            try:
                with open(module, "r+b"):
                    pass
            except PermissionError:
                raise RuntimeError(str(module) + " is in use. Close ComfyUI, then run the installer again.")
        shutil.copytree(build, backup)
    return backup


def _restore_build(backup):
    build = NODE_DIR / "build"
    shutil.rmtree(build, ignore_errors=True)
    if backup.is_dir():
        shutil.copytree(backup, build)


def _install(env):
    env = env.copy()
    vcpkg, installed = _install_cpu_dependencies(env)
    _run([sys.executable, "-m", "pip", "install", "--no-build-isolation", "-r", "requirements-runtime.txt", "-r", "requirements-build.txt"], env)
    pybind = subprocess.check_output([sys.executable, "-m", "pybind11", "--cmakedir"], text=True, env=env).strip()
    # A compiler upgraded in place must not inherit objects from an older run.
    cpu_identity = dict(python=sys.version, executable=sys.executable,
                        compiler=file_identity(shutil.which('cl.exe', path=env['PATH'])),
                        linker=file_identity(shutil.which('link.exe', path=env['PATH'])),
                        cmake=file_identity(shutil.which('cmake.exe', path=env['PATH'])),
                        python_headers=tree_identity(Path(sys.base_prefix) / 'Include'),
                        python_library=file_identity(Path(sys.base_prefix) / 'libs/python312.lib'),
                        pybind=tree_identity(Path(pybind).parents[2] / 'include'),
                        dependencies=tree_identity(installed / 'x64-windows'),
                        toolchain=file_identity(vcpkg / 'scripts/buildsystems/vcpkg.cmake'),
                        cmake_source=file_identity(NODE_DIR / 'CMakeLists.txt'),
                        sources=tree_identity(NODE_DIR / 'native'),
                        configuration='Release',
                        flags={key: env.get(key, '') for key in ('CL', '_CL_', 'LINK', 'CXXFLAGS', 'CFLAGS', 'LDFLAGS', 'INCLUDE', 'LIB')})
    cpu_build = cache_directory(NODE_DIR / '.build/cpu', cpu_identity)
    print('[Installer] CPU Build Cache:', cpu_build, flush=True)
    _run(["cmake", "-S", ".", "-B", cpu_build, "-G", "Ninja",
          "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_COMPILER=cl.exe",
          "-DCMAKE_TOOLCHAIN_FILE=" + str(vcpkg / "scripts/buildsystems/vcpkg.cmake"),
          "-DVCPKG_MANIFEST_MODE=OFF", "-DVCPKG_INSTALLED_DIR=" + str(installed),
          "-DVCPKG_TARGET_TRIPLET=x64-windows", "-DPython3_EXECUTABLE=" + sys.executable,
          "-Dpybind11_DIR=" + pybind], env)
    _run(["cmake", "--build", cpu_build, "--target", "wtivo_core", "wtivo_vdb", "--parallel", env["MAX_JOBS"]], env)
    dlls = list((installed / "x64-windows/bin").glob("*.dll"))
    if not dlls:
        raise RuntimeError("WTiVo CPU dependency DLLs were not found in the vcpkg installation.")
    for dll in dlls:
        shutil.copy2(dll, NODE_DIR / "build" / dll.name)
    _run([sys.executable, "scripts/build_gpupr_hip.py"], env)
    _run([sys.executable, "scripts/verify_hip_graph.py"], env)
    _run([sys.executable, "wtivo.py", "--help"], env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Check prerequisites without compiling or installing.")

    args = parser.parse_args()
    import torch
    if not torch.version.hip:
        raise RuntimeError("Use the ROCm PyTorch environment that runs ComfyUI.")
    if not torch.cuda.is_available():
        raise RuntimeError("The ROCm GPU is unavailable in this PyTorch environment.")
    print("[Installer] Python:", sys.executable)
    print("[Installer] PyTorch:", torch.__version__, "HIP:", torch.version.hip)
    env = _native_environment()
    _gpu_build_architectures(env, torch)
    if args.check:
        print("[Installer] Prerequisites checked; no modules were compiled or installed.")
        return 0
    backup = _snapshot_build()
    try:
        _install(env)
    except BaseException:
        print("[Installer] Installation failed; restoring the previously installed modules.", flush=True)
        _restore_build(backup)
        raise
    print("[Installer] Installation completed. If installing a node group, wait for all installers before restarting ComfyUI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
