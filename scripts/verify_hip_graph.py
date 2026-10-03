# SPDX-License-Identifier: GPL-3.0-or-later
"""Small HIP graph-cut validation against exhaustive reduced cut costs."""
import os, sys, itertools
from pathlib import Path
root=Path(__file__).resolve().parents[1]
handles=[]
if os.name=='nt':
    for d in (root/'build',root/'.deps/vcpkg/installed/x64-windows/bin'):
        if d.exists(): handles.append(os.add_dll_directory(str(d)))
sys.path.insert(0,str(root/'build'))
import numpy as np
import torch
if torch.version.hip is None or not torch.cuda.is_available(): raise RuntimeError('ROCm GPU required')
import wtivo_gpupr as gpu

# Exercise the no-free-node return path (must return all 18 metrics).
v=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float64)
t=np.array([[0,1,2,3]],dtype=np.int32)
n=np.full((1,4),-1,dtype=np.int32)
for value in (0,1):
    result=gpu.graph_cut_fast(v,torch.tensor(t,device='cuda'),torch.tensor(n,device='cuda'),np.array([value],dtype=np.uint8),1.0,1,32,10000,8)
    assert len(result)==18 and np.array_equal(result[0],[value])

rng=np.random.default_rng(143)
pts=np.concatenate([v, rng.uniform(.12,.36,(12,3))])
try:
    import wtivo_core as core
except ImportError:
    # Test-only fallback: validates the HIP solver while CGAL is being built.
    from scipy.spatial import Delaunay
    triangulation=Delaunay(pts)
    vertices,tets,neighbors=pts,triangulation.simplices,triangulation.neighbors[:,::-1]
else:
    vertices,tets,neighbors=core.tetrahedralize_neighbors(np.ascontiguousarray(pts),2)
vertices=np.ascontiguousarray(vertices,dtype=np.float64); tets=np.ascontiguousarray(tets,dtype=np.int32); neighbors=np.ascontiguousarray(neighbors,dtype=np.int32)
interior=np.flatnonzero((neighbors>=0).all(axis=1))
if len(interior)<2: raise RuntimeError('Test fixture has insufficient interior tetrahedra')
free=interior[:8]
labels=np.zeros(len(tets),dtype=np.uint8)
labels[(neighbors<0).any(axis=1)]=1
labels[free]=1
fixed=labels.copy()
lut=((0,1,2),(0,1,3),(0,2,3),(1,2,3))
for fill in (0.0,1.0,20.0):
    edges=[]
    for i,row in enumerate(neighbors):
        for slot,j in enumerate(row):
            if j<0 or j<=i or (i not in free and j not in free): continue
            a,b,c=vertices[tets[i,list(lut[slot])]]
            area=float(np.float32(1e7*.5*np.linalg.norm(np.cross(b-a,c-a))))
            capacity=int(area if labels[i]!=labels[j] else area*(1+fill))
            edges.append((i,j,capacity))
    def cost(partition): return sum(cap for i,j,cap in edges if partition[i]!=partition[j])
    best=min(cost(np.array([assignment[list(free).index(i)] if i in free else fixed[i] for i in range(len(tets))])) for assignment in itertools.product((0,1),repeat=len(free)))
    for steps in (1,8):
        result=gpu.graph_cut_fast(vertices,torch.tensor(tets,device='cuda'),torch.tensor(neighbors,device='cuda'),labels,fill,2,32,10000,steps)
        output=np.asarray(result[0])
        assert len(result)==18
        assert np.array_equal(output[np.setdiff1d(np.arange(len(tets)),free)],fixed[np.setdiff1d(np.arange(len(tets)),free)])
        assert cost(output)==best,(fill,steps,cost(output),best)
        assert int(result[1])==best,(result[1],best)
        surface_v,surface_f=gpu.surface_extraction_topology_cuda(output,vertices,torch.tensor(tets,device='cuda'),torch.tensor(neighbors,device='cuda'),2)
        assert np.asarray(surface_f).ndim==2 and np.asarray(surface_f).shape[1]==3
        def face_key(points): return tuple(sorted(tuple(row) for row in points))
        expected={}
        for i,row in enumerate(neighbors):
            if output[i]!=0: continue
            for slot,j in enumerate(row):
                if j<0 or output[j]!=0:
                    face=vertices[tets[i,list(lut[slot])]]
                    expected[face_key(face)]=vertices[tets[i,3-slot]]
        actual=np.asarray(surface_v)[np.asarray(surface_f)]
        assert len(actual)==len(expected)
        assert {face_key(face) for face in actual}==set(expected)
        for a,b,c in actual:
            opposite=expected[face_key(np.array([a,b,c]))]
            assert np.dot(np.cross(b-a,c-a),opposite-a)<0,'face is oriented into the inside tetrahedron'
torch.cuda.synchronize()
print('PASS: degenerate API, tiny reduced min-cut vs exhaustive oracle, scheduling variants and extraction smoke. Full meshes/VDB still need testing.')
