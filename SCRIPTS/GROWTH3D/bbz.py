from dolfin import *
import numpy as np
import pickle

## Number of elements - same as mesh setup
Nx=40
Ny=20
Nz=41

## Dimensions - same as in mesh setup
Ly=2.0
Lx=1.0 #*Ly
Lz=0.1


###### define file and boudary surfaces - subdomains
Ho=1.2*(Lz/Nz) # depth of the file
dx1=Lx/10

## read mesh setup file
mesh = Mesh()
f = HDF5File(mesh.mpi_comm(), "meshq.hdf5", 'r')
f.read(mesh, "mesh", False)

materials = MeshFunction("size_t", mesh,mesh.topology().dim())
film=CompiledSubDomain("x[2]>=Rcut",Rcut=Lz-Ho) #,size=1.0

materials.set_all(0)
film.mark(materials,1)
#f = HDF5File(mesh.mpi_comm(), "meshq.hdf5", 'r')
#f.read(materials, "materials")

bmats=File("matx.pvd")
bmats << materials


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

V = VectorFunctionSpace(mesh,"CG",1)

Pz = Expression((0.0,0.0,"pz"),pz=0,degree=1)

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

###########################################
# Define functions

du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration
B  = Constant((0.0, -0.0,0.0))  # Body force per unit volume
##############
# Elasticity parameters
E1, nu1 = 1, 0.48
E2 = 20*E1
E3 = 20*E1
nu2=nu1
nu3=nu1

## Elasticity parameters 1 BULK-0
mu1, lda1 = Constant(E1/(2*(1 + nu1))), Constant(E1*nu1/((1 + nu1)*(1 - 2*nu1)))
## Elasticity parameters 3 FILM - 2
mu3, lda3 = Constant(E3/(2*(1 + nu3))), Constant(E3*nu3/((1 + nu3)*(1 - 2*nu3)))


d = u.geometric_dimension()
I = Identity(d)             # Identity tensor

#Growth Film
dgnx=1.0
dgny=1.0
dgnz=1.0

Fgf = Expression( (("dgnx",0.0,0.0),(0.0,"dgny",0.0),(0.0,0.0,"dgnz")), dgnx=dgnx, dgny=dgny, dgnz=dgnz, degree=1)
Fgfinv = inv(Fgf)
Jgf = det(Fgf)

#Growth Sub
Fgs = Expression ((("dgn", 0.0, 0.0), (0.0, "dgn2", 0.0), (0.0,0.0,"dgn3")),dgn=dgnx,dgn2=dgny,dgn3=dgny,degree=1)
Fgsinv = inv(Fgs)
Jgs = det(Fgs) #Fgs ** 3

Fgb= I
Fgbinv=inv(Fgb)
Jgb = det(Fgb)

#############
# set all the tensors, growth, elastic deformation gradients and stress

## Kinematics
F = I + grad(u)             # Deformation gradient

##FILM
Fef = F * Fgfinv              # Elastic deformation gradient
Cef = Fef.T * Fef              # Elastic right Cauchy-Green tensor

## Invariants of deformation tensors
Icef = variable(tr(Cef))
Jef  = variable(det(Fef))

## Stored strain energy density (compressible neo-Hookean model)
psif = (mu3/2)*(Icef - 3) - mu3*ln(Jef) + (lda3/2)*(ln(Jef))**2

## Elastic second Piola-Kirchhoff stress tensor
Sef = 2*diff(psif, Icef)*I + Jef*Jef*diff(psif, Jef)*inv(Cef)

## Total second Piola-Kirchhoff stress
Sf = Jgf*Fgfinv * Sef * Fgfinv

## First Piola-Kirchhoff stress tensor
Pf = F*Sf

## SUBSTRATE
Fes = F * Fgsinv              # Elastic deformation gradient
Ces = Fes.T * Fes              # Elastic right Cauchy-Green tensor

# Invariants of deformation tensors
Ices = variable(tr(Ces))
Jes  = variable(det(Fes))

# Stored strain energy density (compressible neo-Hookean model)
psis = (mu1/2)*(Ices - 3) - mu1*ln(Jes) + (lda1/2)*(ln(Jes))**2

# Elastic second Piola-Kirchhoff stress tensor
Ses = 2*diff(psis, Ices)*I + Jes*Jes*diff(psis, Jes)*inv(Ces)

# Total second Piola-Kirchhoff stress
Ss = Jgs*Fgsinv * Ses * Fgsinv

# First Piola-Kirchhoff stress tensor
Ps = F*Ss

#######################
## write variation form

dx = Measure('dx', domain=mesh, subdomain_data=materials)
dx = dx(degree=4)
ds = Measure("ds", subdomain_data=boundary_parts)
ds=ds(degree=4)

F = inner(Ps, grad(v))*dx(0) + inner(Pf, grad(v))*dx(1)-dot(Pz,v)*ds(3)

# Compute Jacobian of F
J = derivative(F, u, du)



################################
## solve

##########
## set up run options
parameters["form_compiler"]["cpp_optimize"]=True
parameters["form_compiler"]["representation"]='uflacs'
parameters['std_out_all_processes'] = False

######

file = File("displacement.pvd");

file << u;
mu=0

d1=0.000
d2=0.000

ky=0.0
kx=0.0

DZ=-0.001
DX=0
DY=0
pzo=0.0
mp=1

muk=0.0
dt=1e-3
t=0.0
T=1.0
dtp=0.01
d1=0.0
d2=0.0
gfx=1.0
gfy=1.0
gsx=1.0
gsy=1.0
gsz=1.0
tp=0.01
muk=0.0

while t<T:

    print(t,tp)
    if t>0.204:
        dt=1e-4
    t+=dt
    d1+=0.0*dt
    d2+=1.0*dt
    gfx+=1.0*dt
    gfy+=1.0*dt
    if t<0.2:
        gsz+=0.1*dt
    if t<0.2:
        muk+=0.1*dt

    Fgf.dgnx = gfx
    Fgf.dgny = gfy

    Fgs.dgn = gsz
    Fgs.dgn2 = gsz
    Fgs.dgn3 = gfx

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
    cb.R=muk

    bcsa =[bcr,bcl,bcb,bcn,bcs]

    solve(F == 0, u , bcsa, J=J, solver_parameters={"newton_solver":{"maximum_iterations":100,
                                                        "relative_tolerance": 1.0e-14,
                                                        "absolute_tolerance": 1.0e-6,"linear_solver": "mumps"}})
    if t>tp:
        file << u;
        tp+=0.01
        print("Write displacement")
