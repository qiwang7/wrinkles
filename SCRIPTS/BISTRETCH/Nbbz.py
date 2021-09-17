from dolfin import *
import numpy as np
import pickle


## Number of elements - same as mesh setup
Nx=20
Ny=40
Nz=41

## Dimensions - same as in mesh setup
Ly=1.0
Lx=2.0
Lz=0.1

###### define file and boudary surfaces - subdomains
Ho=1.2*(Lz/Nz) # depth of the file
dx1=Lx/10

## read mesh setup file
mesh = Mesh()
f = HDF5File(mesh.mpi_comm(), "meshq.hdf5", 'r')
f.read(mesh, "mesh", False)

## Mark boundary subdomians - material
materials=MeshFunction("size_t",mesh,mesh.topology().dim())
film=CompiledSubDomain("x[2]>=Rcut",Rcut=Lz-Ho)
materials.set_all(0)
film.mark(materials,1)

## save the material domains
bmats=File("matx.pvd")
bmats << materials

## set the boundaries at the bottom and all the lateral surfaces
left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
bott =  CompiledSubDomain("near(x[2], side) && on_boundary", side = 0.0)
right = CompiledSubDomain("near(x[0], side) && on_boundary", side = Lx)
southx= CompiledSubDomain("near(x[1], side) && on_boundary", side = 0.0)
northx= CompiledSubDomain("near(x[1], side) && on_boundary", side = Ly)

boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)
bott.mark(boundary_parts, 3)
left.mark(boundary_parts, 1)
right.mark(boundary_parts, 2)
southx.mark(boundary_parts, 4)
northx.mark(boundary_parts, 5)

# save boundary file
bmark = File("bmarks_mark.pvd")
bmark << boundary_parts

## impose domain deformation
cl = Expression(("x0","ky*x[1]","R*(x[1])*(x[1]-z0)"),x0 = 0.0, ky = 0.0, z0=Ly,R=-0.00,degree=1)
cr = Expression(("x0","ky*x[1]","R*(x[1])*(x[1]-z0)"),x0 = 0.0, ky = 0.0, z0=Ly,R=-0.00,degree=1)
cb = Expression(("kx*x[0]","ky*x[1]","R*(x[1])*(x[1]-z0)"),kx= 0.0, ky = 0.0, z0=Ly,R=-0.00,degree=1)
cs = Expression(("kx*x[0]","ky*x[1]","R*(x[1])*(x[1]-z0)"),kx= 0.0, ky = 0.0, z0=Ly,R=-0.00,degree=1)
cn = Expression(("kx*x[0]","y0","R*(x[1])*(x[1]-z0)"),kx= 0.0, y0 = 0.0, z0=Ly,R=-0.00,degree=1)

bcl = DirichletBC(V, cl, left)
bcr = DirichletBC(V, cr, right)
bcb = DirichletBC(V, cb, bott)
bcs = DirichletBC(V, cs, southx)
bcn = DirichletBC(V, cn, northx)

########
# Solution
V = VectorFunctionSpace(mesh,"CG",1)
# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration
B  = Constant((0.0, -0.0,0.0))  # Body force per unit volume

##############
# Elasticity parameters
E1, nu1 = 1e6, 0.48
E3 = 25*E1
nu3=nu1

## Elasticity parameters 1 BULK-0
mu1, lda1 = Constant(E1/(2*(1 + nu1))), Constant(E1*nu1/((1 + nu1)*(1 - 2*nu1)))
## Elasticity parameters 3 FILM - 2
mu3, lda3 = Constant(E3/(2*(1 + nu3))), Constant(E3*nu3/((1 + nu3)*(1 - 2*nu3)))

d = u.geometric_dimension()
I = Identity(d)             # Identity tensor

## Kinematics
F = I + grad(u)             # Deformation gradient
## FILM
C = F.T * F              # Elastic right Cauchy-Green tensor
## Invariants of deformation tensors
I = variable(tr(C))
Je  = variable(det(F))

## Stored strain energy density (compressible neo-Hookean model)
psif = (mu3/2)*(I - 3) - mu3*ln(Je) + (lda3/2)*(ln(Je))**2
psis = (mu1/2)*(I - 3) - mu1*ln(Je) + (lda1/2)*(ln(Je))**2

##
dx = Measure('dx', domain=mesh, subdomain_data=materials)
dx = dx(degree=4)
ds = Measure("ds", subdomain_data=boundary_parts)
ds = ds(degree=4)

Pi=psif*dx(1)+psis*dx(0) # the energy
F=derivative(Pi,u,v)
## Compute Jacobian of F
J = derivative(F, u, du)

##########
## set up run options
parameters["form_compiler"]["cpp_optimize"]=True
parameters["form_compiler"]["representation"]='uflacs'
parameters['std_out_all_processes'] = False

##########
## solver

file = File("displacement.pvd");
file << u;
mu=0

d1=0.000
d2=0.000
ky=0.0
kx=0.0

DZ=-0.001
DX=0
DY=0 # step size
pzo=0.0
mp=1 # number of steps to store the displacement

for j in range(100):
    print(j)
    if j <5:
        DY=0.05
        mp=1
    elif j<10: # j>=5 and j<10
        DY=0.01
        mp=5
    else: # j>=10:
        DX=1e-3
        mp=10
    DX=(Lx*DY)/(Ly-DY)
    print(DX,DY)
    d2+=(-1.0*DY)
    d1+=(1.0*DX)
    ky=d2/Ly
    kxb=d1/Lx
    cr.x0=d1
    cr.ky=ky
    cl.ky=ky
    cb.kx=kxb
    cb.ky=ky
    cn.y0=d2
    cn.kx=kxb
    cs.kx=kxb
    bcsa =[bcr,bcl,bcb,bcn,bcs]#,bcb]#S,bcrs,bcrn] # [bclu,bcld,bcru,bcrd,bcl,bcr]
    solve(F == 0, u , bcsa, J=J, solver_parameters={"newton_solver":{"maximum_iterations":100,
                                                        "relative_tolerance": 1.0e-14,
                                                        "absolute_tolerance": 1.0e-6,
                                                        "linear_solver": "mumps"}
                                                        })
    ## write displacement
    if j%mp==0:
        file << u;
