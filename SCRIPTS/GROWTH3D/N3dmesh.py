from dolfin import *
import numpy as np

## Number of elements
Nx=40
Ny=20
Nz=41

## Dimensions
Ly=2.0
Lx=1.0#*Ly
Lz=0.1

mesh = UnitCubeMesh.create(Nx, Ny, Nz, CellType.Type.hexahedron)
# mesh size is smaller near x=y=0
mesh.coordinates()[:, 0] = mesh.coordinates()[:, 0]*Lx
mesh.coordinates()[:, 1] = mesh.coordinates()[:, 1]*Ly
mesh.coordinates()[:, 2] = mesh.coordinates()[:, 2]*Lz

f = HDF5File(mesh.mpi_comm(), "meshq.hdf5", 'w') # store mesh setup
f.write(mesh, "mesh")
