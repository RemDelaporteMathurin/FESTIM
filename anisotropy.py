from mpi4py import MPI
import dolfinx
import ufl
import festim as F


my_model = F.HydrogenTransportProblem()

nx = ny = 20
my_model.mesh = F.Mesh(
    dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, nx, ny),
)


class AnisotropicMaterial(F.Material):
    def __init__(self, D):
        self.D = D

    # need to overwrite this method since it calls D_0 and E_D
    def get_diffusion_coefficient(self, mesh, temperature, species=None):
        return self.D


D = ufl.as_tensor([[100, 0], [0, 1]])
my_mat = AnisotropicMaterial(D=D)

my_subdomain = F.VolumeSubdomain(id=1, material=my_mat)
boundary = F.SurfaceSubdomain(id=1)
my_model.subdomains = [my_subdomain, boundary]

mobile_H = F.Species("H")
my_model.species = [mobile_H]

my_model.temperature = 500.0

my_model.boundary_conditions = [
    F.DirichletBC(subdomain=boundary, value=0, species=mobile_H)
]

my_model.sources = [F.ParticleSource(volume=my_subdomain, species=mobile_H, value=1)]

my_model.exports = [F.XDMFExport("mobile_concentration.xdmf", field=mobile_H)]

my_model.settings = F.Settings(
    atol=1e-10,
    rtol=1e-10,
    max_iterations=30,
    transient=False,
)

my_model.initialise()
my_model.run()
