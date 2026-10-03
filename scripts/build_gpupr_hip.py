# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the HIP solver with the user's existing Windows ROCm interpreter."""
from __future__ import annotations
import importlib.util, json, os, shutil, sys
from pathlib import Path

def main():
    if os.name != 'nt':
        raise SystemExit('WTiVo currently supports Windows x64.')
    import torch
    if torch.version.hip is None:
        raise SystemExit('Use your ROCm Torch interpreter; NVIDIA/CPU Torch is unsupported by this builder.')
    if not torch.cuda.is_available():
        raise SystemExit('ROCm Torch cannot see the GPU.')
    sdk = os.environ.get('ROCM_HOME') or os.environ.get('ROCM_PATH') or os.environ.get('HIP_PATH')
    if not sdk:
        for name in ('_rocm_sdk_core', '_rocm_sdk_devel'):
            spec = importlib.util.find_spec(name)
            if spec and spec.origin:
                sdk = str(Path(spec.origin).parent)
                break
    if not sdk:
        raise SystemExit('Initialize the matching ROCm SDK through your existing ComfyUI launcher first.')
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
    build = root / '.build' / 'gpupr-hip'
    build.mkdir(parents=True, exist_ok=True)
    out = root / 'build'
    out.mkdir(exist_ok=True)
    print('Python:', sys.executable, 'Torch:', torch.__version__, 'HIP:', torch.version.hip, 'SDK:', sdk, flush=True)
    module = ext.load(name='wtivo_gpupr', sources=[str(root / 'native/gpupr/wtivo_gpupr_bindings.cpp'), str(root / 'native/gpupr/gpu_push_relabel_fast.cu')], build_directory=str(build), with_cuda=True, extra_cflags=['-O2'], extra_cuda_cflags=['-O3', '-std=c++20', '-fno-fast-math'], verbose=True)
    for symbol in ('graph_cut_fast', 'surface_extraction_topology_cuda'):
        if not hasattr(module, symbol):
            raise RuntimeError('Missing ' + symbol)
    shutil.copy2(module.__file__, out / 'wtivo_gpupr.pyd')
    (out / 'hip-build-info.json').write_text(json.dumps(dict(python=sys.executable, torch=torch.__version__, hip=torch.version.hip, sdk=sdk), indent=2))
    print('Built HIP module. Run scripts/verify_hip_graph.py before using it on real meshes.')
if __name__ == '__main__':
    main()
