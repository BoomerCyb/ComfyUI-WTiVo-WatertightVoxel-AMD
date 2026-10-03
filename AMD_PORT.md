# AMD changes

GPU runtime calls use HIP. Device guards and synchronization are retained.
The empty graph return and owned tetrahedralization vertex fixes are preserved.
CPU modules continue to use CGAL, Eigen, oneTBB and OpenVDB.

Installation uses ComfyUI's ROCm Python through install.py. Native extensions
use PyTorch HIP device properties to target discrete GPUs by default, while
respecting PYTORCH_ROCM_ARCH. Integrated-only systems target their available GPUs.
Tested on RX 9070 XT; other supported ROCm devices require validation.
Original licenses, algorithms and node registrations are preserved.
