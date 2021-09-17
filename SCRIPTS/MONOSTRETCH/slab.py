from dolfin import *
import sys
sys.path.append("../../../../FMECHS/")
import fenicsmechanics as fm
import numpy as np

## Number of elements
Nr=100
Nt=100
Nz=1

## Dimensions
Ly=5
Lx=2*Ly
Lz=1.00

## Transform of the unit mesh into a slender rectangle
#Lzn=0.001
Lzn=0.005

meshname=BoxMesh(Point(0.0,0.0,0.0),Point(Lx,Ly,Lz),Nr,Nt,Nz)#
x=meshname.coordinates()[:,0]
y=meshname.coordinates()[:,1]
z=meshname.coordinates()[:,2]
def transmesh(x,y,z,Lzn):#
    return [x,y,Lzn*z]
qn=transmesh(x,y,z,Lzn)
qnx=np.array(qn).transpose()
meshname.coordinates()[:]=qnx
tol=1E-14

# Define the clamped boundaries

lxa=1.0*Lx/20

xlc =  CompiledSubDomain("(near(x[2], side) && on_boundary) && x[0]<lxa", side = 0.0, lxa=lxa)
xld =  CompiledSubDomain("(near(x[2], side) && on_boundary) && x[0]<lxa", side = Lzn, lxa=lxa)

xrc = CompiledSubDomain("(near(x[2], side) && on_boundary) && x[0]>lxb", side = 0.0, lxb=Lx-lxa)
xrd = CompiledSubDomain("(near(x[2], side) && on_boundary) && x[0]>lxb", side = Lzn, lxb=Lx-lxa)

boundary_parts = MeshFunction("size_t", meshname, meshname.topology().dim() - 1)
boundary_parts.set_all(0)

xlc.mark(boundary_parts,1)
xld.mark(boundary_parts,2)

xrc.mark(boundary_parts,3)
xrd.mark(boundary_parts,4)

## save boundary marks as a file
bds=File("bmarksmarks.pvd")
bds << boundary_parts

######################
# solve
######################

## initial condition
cr = Expression(("x0","kr*x[1]","z0"),x0 = 0.0, kr = 0.0, z0=0.0000,degree=1)
cl = Expression(("x0","kl*x[1]","z0"),x0 = 0.0, kl = 0.0, z0=0.0,degree=1)

## set mechanical problem
material = {
    'type': 'elastic',
    'const_eqn': 'neo_hookean',
    'incompressible': True,
    'kappa': 10e9, # Pa
    'mu': 1.5e6 # Pa
}

mesh = {
    'mesh_file': meshname,
    'boundaries': boundary_parts
}

formulation = {
    'element': 'p2-p1',
    'domain': 'lagrangian',
    'bcs': {
        'dirichlet': {
            #'displacement': [cr,cl,cu,cd],
            'displacement': [cl,cl,cr,cr],
            'regions': [1,2,3,4],
            }
    }
}

config = {
    'material': material,
    'mesh': mesh,
    'formulation': formulation
}


filen = "2ddisplacement.pvd" # store solution
problem = fm.SolidMechanicsProblem(config)
solver = fm.SolidMechanicsSolver(problem, fname_disp=filen)
solver.set_parameters(linear_solver="mumps")
solver.set_parameters(newton_maxIters=500)
solver.set_parameters(newton_abstol=1e-7)
solver.set_parameters(newton_reltol=1e-6)

## updating the values of the boundary condition expression

d1=0.000
d2=0.000

ky=0.0
kx=0.0

for j in range(4000):
    # smaller step size later
    ## d1 step size
    if j<50 or j> 8000:
        DX=0.01
    else:
        DX=0.001
    ## d2 step size
    if j<5 or j>800:
        DY=0.001
    else:
        DY=1e-9
    ## update
    d2+=DY
    print(d1)
    d1+=DX
    cr.x0=d1
    ky=d2/Ly
    cl.kl=ky
    cr.kr=ky
    solver.full_solve()
