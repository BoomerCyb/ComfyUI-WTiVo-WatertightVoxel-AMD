# ComfyUI-WTiVo-WatertightVoxel-AMD

## AMD / ROCm

The installer uses ComfyUI's Python and stops if setup fails. When installing through EZi, wait for the entire node group to complete before restarting.

This fork keeps the original node interface and adds native HIP support.
Use the ROCm PyTorch installation that runs ComfyUI and a matching HIP SDK.
The source does not select a card model or impose a gfx1201 target. Native
extensions target visible discrete AMD GPUs by default, excluding integrated GPUs
when a discrete GPU is available. Set PYTORCH_ROCM_ARCH to override the targets.
Integrated-only systems remain supported; PyTorch supplies the compiler flags.
An installed binary still needs to match its GPU target, Python and Torch runtime.
Hardware support depends on ROCm/PyTorch; validation here covers RX 9070 XT.

Run `install_requirements.bat` with ComfyUI closed to build/install the native components.
The installer first installs Eigen3, CGAL, OpenVDB and TBB from `vcpkg.json`
into this node's `.deps/vcpkg_installed` folder, then builds against that same folder.
It uses a complete existing vcpkg checkout or downloads and bootstraps a local one.
The first dependency build can take considerable time. The matching HIP SDK and
Visual Studio C++ Build Tools/Windows SDK are still required.
For prerequisites and manual commands, see [COMFYUI_ROCM_BUILD_GUIDE.md](COMFYUI_ROCM_BUILD_GUIDE.md).

ComfyUI AMD installer: [BoomerCyb/ComfyUI-Easy-Install-AMD](https://github.com/BoomerCyb/ComfyUI-Easy-Install-AMD).


A high-performance, in-process watertight remeshing node for ComfyUI. WTiVo converts defective, non-manifold, or open 3D triangle meshes (such as raw outputs from AI 3D generators like TRELLIS) into dense, closed, manifold meshes using sparse voxel fields, tetrahedral cell cuts, HIP graph optimization, and manifold contouring.

## ⚡ Key Highlights & Changes

* **Native ComfyUI Custom Node:** Converted from a standalone CLI script into a native, in-process ComfyUI node (`WTiVo - Mesh Watertight`), executing directly on native `MESH` objects.
* **2K / High-Detail Reconstruction:** WTiVo can now process **2K meshes with significantly more geometric detail** by increasing the input resolution, final resolution, and proxy-point budget.
* **Higher Proxy-Point Quality:** The previous `12 million` proxy-point workflow can be increased to approximately **25 million proxy points** for high-detail 2K reconstruction.
* **More Proxy Points = More Detail:** Increasing the proxy-point budget gives the reconstruction more geometric information to work with. For detailed meshes, higher proxy-point counts can improve preservation of small features, edges, and surface detail.
* **16 GB+ VRAM Recommended for High Detail:** A GPU with **16 GB VRAM or more** is recommended when using 2K resolution together with approximately 25 million proxy points.
* **AMD Native Backend:** The installer builds the CPU and HIP extensions for ComfyUI's Python and Torch environment.
* **In-Memory Zero-IPC Pipeline:** Automatically unloads active upstream models before processing and passes native PyTorch tensors and NumPy arrays directly in memory—no temporary GLB/OBJ files or subprocess bridges.
* **HIP-Accelerated Graph Cut:** Leverages multi-discharge Push-Relabel HIP graph optimization and CGAL Delaunay tetrahedralization for fast watertight surface extraction.
* **Exact Edge Validation:** Performs topology validation using exact edge-degree checks to verify the resulting mesh topology.

---

## 🖼️ Examples

### 🔥 2K High-Detail Reconstruction

WTiVo can reconstruct meshes at **2K resolution** while preserving significantly more surface detail. Increasing the proxy point count gives the reconstruction more geometry to work with, allowing fine details and complex shapes to be preserved.

**2K Example 1**

![WTiVo 2K Example](https://github.com/Mstafa-awad/WTiVo-WatertightVoxel-ComfyuiNode/blob/main/Doc/2K.png)

**2K Example 2**

![WTiVo 2K High-Detail Example](https://github.com/Mstafa-awad/WTiVo-WatertightVoxel-ComfyuiNode/blob/main/Doc/2k-1.png)

> **Recommended for 2K:** `input_res = 2048`, `final_res = 2048`, and around `25,000,000` proxy points. Higher proxy point counts can preserve more detail, but require additional VRAM and processing time.

---

### 💧 Watertight Reconstruction

WTiVo rebuilds the mesh topology to produce a **closed, manifold, watertight surface**, making the resulting mesh suitable for downstream processing such as decimation, UV unwrapping, baking, rigging, and game-ready asset workflows.

![WTiVo Watertight Example](https://github.com/Mstafa-awad/WTiVo-WatertightVoxel-ComfyuiNode/blob/main/Doc/watertight.png)

---

## 🚀 New: 2K High-Detail Mesh Support

WTiVo is no longer limited to the previous 1536-resolution / 12-million-proxy-point workflow.

For **2K meshes containing a lot of geometric detail**, increase the reconstruction settings:

```text
input_res      = 2048
final_res      = 2048
proxy_points   = 25,000,000
```

### Recommended configurations

| Workflow       | Input Resolution | Final Resolution | Proxy Points | VRAM                    |
| :------------- | :--------------: | :--------------: | :----------: | :---------------------- |
| Standard       |      `1536`      |      `1536`      |     `12M`    | ~8 GB+                  |
| High Detail    |      `1536`      |      `1536`      |     `20M`    | Higher VRAM recommended |
| 2K High Detail |      `2048`      |      `2048`      |     `25M`    | **16 GB+ recommended**  |

These are practical starting points rather than hard limits.

### More proxy points can preserve more detail

The proxy-point budget is an important part of the reconstruction quality.

For example:

```text
12,000,000 proxy points
        ↓
Good for standard workflows

20,000,000 proxy points
        ↓
Higher geometric budget

25,000,000+ proxy points
        ↓
Designed for high-detail / 2K workflows
```

If your GPU has enough VRAM, increasing the number of proxy points can provide the reconstruction with more geometric information and can help preserve fine details.

**For GPUs with 16 GB VRAM or more, around 25 million proxy points is recommended for 2K high-detail workflows.**

> **Important:** More proxy points require more memory. If you run out of VRAM, reduce `proxy_points` first or lower the reconstruction resolution.

---

## Installation

1. Place this repository in `ComfyUI/custom_nodes/ComfyUI-WTiVo-WatertightVoxel-AMD`.
2. Close ComfyUI and run `install_requirements.bat` using ComfyUI's Python.
3. Restart ComfyUI after installation completes. With the EZi group add-on, wait for all five nodes to finish.

You can also install this node through **Easy Menu → Add-ons → BoomerCyb WTiVo AMD Nodes** in [ComfyUI-Easy-Install-AMD](https://github.com/BoomerCyb/ComfyUI-Easy-Install-AMD).

## ⚙️ Node Parameters

| Parameter               | Type      |   Default  | Description                                                                                                                                             |
| :---------------------- | :-------- | :--------: | :------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `mesh`                  | **MESH**  |      —     | Input ComfyUI native mesh object, e.g. from `VaeDecodeShapeTrellis`.                                                                                    |
| `input_res`             | **INT**   |   `1536`   | Resolution for the initial thick UDF sparse field and graph-labeling pass. Increase to `2048` for 2K workflows.                                         |
| `final_res`             | **INT**   |   `1536`   | Resolution for the final signed OpenVDB field and FaithC reconstruction. Increase to `2048` for 2K workflows.                                           |
| `proxy_points`          | **INT**   | `12000000` | QEF proxy-point budget sent to CGAL for tetrahedralization. Around `25,000,000` is recommended for high-detail 2K meshes when enough VRAM is available. |
| `proxy_eps_scale`       | **FLOAT** |    `1.0`   | Inset scale for graph proxy geometry relative to resolution.                                                                                            |
| `proxy_feature_weight`  | **FLOAT** |    `1.5`   | Priority weight for preserving sharp corners, edges, and creases.                                                                                       |
| `lambda_fill`           | **FLOAT** |   `20.0`   | Regularization factor for tetrahedral cell-cut fill.                                                                                                    |
| `thin_iso_vox`          | **FLOAT** |    `0.0`   | Surface offset in voxel units for the final signed field.                                                                                               |
| `faithc_component_mode` | **ENUM**  |  `largest` | Component handling: `auto`, `keep_all`, or `largest`.                                                                                                   |
| `threads`               | **INT**   |  *Max CPU* | Number of logical CPU threads allocated for parallel operations.                                                                                        |

---

## 🔄 Workflow Integration

```text
[ VaeDecodeShapeTrellis ]
          │
          │ MESH
          ▼
[ WTiVo - Mesh Watertight ]
          │
          │ Watertight MESH
          ▼
[ Preview 3D / Save 3D / Exporter ]
```

### Standard workflow

```text
input_res      = 1536
final_res      = 1536
proxy_points   = 12,000,000
```

### 2K high-detail workflow

```text
input_res      = 2048
final_res      = 2048
proxy_points   = 25,000,000
```

For GPUs with **16 GB VRAM or more**, the 2K configuration provides a substantially larger reconstruction budget for detailed geometry.

> **Note:** WTiVo completely rebuilds topology to create a watertight surface. Input UVs, textures, and material maps are not preserved. Ensure batch size is set to `1`.

---

## 🧠 Quality vs Memory

WTiVo's reconstruction quality is affected by both **resolution** and **proxy-point density**.

A simplified way to think about it:

```text
Higher input resolution
          +
Higher final resolution
          +
More proxy points
          ↓
More geometric information
          ↓
Potentially more preserved detail
          ↓
Higher VRAM / RAM usage
```

For example:

```text
1536 / 1536 / 12M
```

is designed around a more moderate memory budget.

Whereas:

```text
2048 / 2048 / 25M
```

is designed for **high-detail 2K reconstruction**.

If your GPU has more available VRAM, increasing the proxy-point count can allow WTiVo to work with a larger geometric budget.

---

## ⚠️ Important Notes

* WTiVo completely rebuilds the mesh topology during watertight reconstruction.
* The resulting mesh is designed to be closed and manifold when reconstruction succeeds.
* Input UVs, textures, and material maps are **not preserved**.
* Use **batch size 1**.
* Higher resolutions and proxy-point counts require significantly more memory.
* **16 GB VRAM or more is recommended for 2K / ~25M proxy-point workflows.**
* If you encounter VRAM or memory limitations, reduce `proxy_points` first, then reduce `input_res` / `final_res`.
* More proxy points generally provide a larger geometric budget and can improve preservation of fine details, but the final result also depends on the input mesh and other reconstruction settings.
* Use the correct `.pyd` build for your GPU generation.

---

## 🔍 Topology Validation

WTiVo performs topology validation on the reconstructed mesh.

The backend checks edge groups to determine whether the resulting surface has:

```text
boundary edges = 0
non-manifold edges = 0
```

A successful watertight result is reported as:

```text
watertight=True
```

This makes WTiVo useful as a preprocessing stage for:

* AI-generated 3D assets
* mesh optimization
* low-poly conversion
* baking
* game-ready asset pipelines
* further remeshing
* 3D printing workflows

---

## ⚖️ License & Attribution

* **License:** GNU General Public License v3.0 or later (**GPL-3.0-or-later**).
* **Attribution:** WTiVo builds upon research and methods from:

  * **CelloCut:** Constructive Watertight Remeshing via Tetrahedral Cell Cuts (Xuan Yang et al.)
  * **FaithC / Faithful Contouring** (Yihao Luo et al.)
  * OpenVDB
  * CGAL
  * oneTBB
  * Eigen
  * PyTorch

---

## AMD Edition Changes - 2026-10-03

- Uses ComfyUI's Python and reports installation failures before restarting.
- Builds native HIP extensions for the active ROCm environment; matching HIP SDK and Visual Studio C++ Build Tools are required.
- Supports group installation through [ComfyUI-Easy-Install-AMD](https://github.com/BoomerCyb/ComfyUI-Easy-Install-AMD).

Original node by [Mstafa-awad / MostAadTech](https://github.com/Mstafa-awad). AMD fork maintained by [BoomerCyb](https://github.com/BoomerCyb). Original license and third-party credits are retained.
