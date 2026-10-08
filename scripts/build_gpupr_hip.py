# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the HIP solver with the user's existing Windows ROCm interpreter."""
from __future__ import annotations
import json, os, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cache import file_identity, tree_identity, cache_directory

def main():
    if os.name != 'nt':
        raise SystemExit('WTiVo currently supports Windows x64.')
    import torch
    if torch.version.hip is None:
        raise SystemExit('Use your ROCm Torch interpreter; NVIDIA/CPU Torch is unsupported by this builder.')
    if not torch.cuda.is_available():
        raise SystemExit('ROCm Torch cannot see the GPU.')
    # Same SDK choice as install.py: the SDK installed with PyTorch first.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from install import _rocm_sdk
    sdk = _rocm_sdk({key.upper(): value for key, value in os.environ.items()})
    os.environ['ROCM_HOME'] = sdk
    os.environ['HIP_PATH'] = sdk
    os.environ.setdefault('MAX_JOBS', '2')
    clang_cl = Path(sdk) / 'lib/llvm/bin/clang-cl.exe'
    if clang_cl.exists():
        os.environ.setdefault('CXX', str(clang_cl))
    from torch.utils import cpp_extension as ext
    print('Ninja:', shutil.which('ninja'), 'CXX:', os.environ.get('CXX'), flush=True)
    if not ext.IS_HIP_EXTENSION:
        raise SystemExit('Installed cpp_extension did not select HIP. No CUDA fallback will be attempted.')
    root = Path(__file__).resolve().parents[1]
    cflags = ['-O2']
    hipflags = ['-O3', '-std=c++20', '-fno-fast-math']
    architectures = os.environ.get('PYTORCH_ROCM_ARCH', '').strip()
    if not architectures:
        architectures = ';'.join(sorted({torch.cuda.get_device_properties(i).gcnArchName.split(':')[0]
                                        for i in range(torch.cuda.device_count())}))
        os.environ['PYTORCH_ROCM_ARCH'] = architectures
    identity = dict(python=sys.executable, python_version=sys.version,
                    torch=torch.__version__, hip=torch.version.hip, sdk=sdk,
                    architectures=architectures, cflags=cflags, hipflags=hipflags,
                    compiler=file_identity(shutil.which(os.environ['CXX']) or os.environ['CXX']),
                    hipcc=file_identity(Path(sdk) / 'bin/hipcc.exe'),
                    compiler_driver=file_identity(Path(sdk) / 'lib/llvm/bin/clang.exe'),
                    torch_headers=tree_identity(Path(torch.__file__).parent / 'include'),
                    # Only the HIP headers reach this build; the full SDK include tree has thousands of
                    # unrelated library headers (hashing them took ~11 s).
                    sdk_headers=tree_identity(Path(sdk) / 'include/hip'),
                    sdk_version=file_identity(Path(sdk) / 'bin/.hipVersion'),
                    torch_libraries=tree_identity(Path(torch.__file__).parent / 'lib'),
                    hip_library=file_identity(Path(sdk) / 'lib/amdhip64.lib'),
                    linker=file_identity(shutil.which('link.exe')),
                    environment={key: os.environ.get(key, '') for key in
                                 ('CL', '_CL_', 'LINK', 'CFLAGS', 'CXXFLAGS', 'LDFLAGS', 'INCLUDE', 'LIB')},
                    python_headers=tree_identity(Path(sys.base_prefix) / 'Include'),
                    sources=tree_identity(root / 'native/gpupr'))
    build = cache_directory(root / '.build/gpupr-hip', identity)
    print('HIP Build Cache:', build, flush=True)
    out = root / 'build'
    out.mkdir(exist_ok=True)
    print('Python:', sys.executable, 'Torch:', torch.__version__, 'HIP:', torch.version.hip, 'SDK:', sdk, flush=True)
    module = ext.load(name='wtivo_gpupr', sources=[str(root / 'native/gpupr/wtivo_gpupr_bindings.cpp'), str(root / 'native/gpupr/gpu_push_relabel_fast.cu')], build_directory=str(build), with_cuda=True, extra_cflags=cflags, extra_cuda_cflags=hipflags, verbose=True)
    for symbol in ('graph_cut_fast', 'surface_extraction_topology_cuda'):
        if not hasattr(module, symbol):
            raise RuntimeError('Missing ' + symbol)
    shutil.copy2(module.__file__, out / 'wtivo_gpupr.pyd')
    # Record provenance only after successful compilation, loading and copying.
    identity['artifact'] = file_identity(out / 'wtivo_gpupr.pyd')
    identity['build_directory'] = str(build)
    record = out / 'hip-build-info.json.tmp'
    record.write_text(json.dumps(identity, indent=2), encoding='utf-8')
    record.replace(out / 'hip-build-info.json')
    print('Built HIP module. Run scripts/verify_hip_graph.py before using it on real meshes.')
if __name__ == '__main__':
    main()
