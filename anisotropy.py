from mpi4py import MPI
import dolfinx
import ufl
import numpy as np

domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 12, 12)


import festim as F

my_mesh = F.Mesh(mesh=domain)

my_model = F.HydrogenTransportProblem()
my_model.mesh = my_mesh


class AnisotropicMaterial(F.Material):
    def __init__(self, D):
        self.D = D

    def get_diffusion_coefficient(self, mesh, temperature, species=None):
        return self.D


D = ufl.as_tensor([[100, 0], [0, 1]])
my_mat = AnisotropicMaterial(D=D)

my_subdomain = F.VolumeSubdomain(id=1, material=my_mat)


class TopSurface(F.SurfaceSubdomain):
    def locate_boundary_facet_indices(self, mesh, fdim):
        indices = dolfinx.mesh.locate_entities_boundary(
            mesh, fdim, lambda x: np.isclose(x[1], 1)
        )
        return indices


class BottomSurface(F.SurfaceSubdomain):
    def locate_boundary_facet_indices(self, mesh, fdim):
        indices = dolfinx.mesh.locate_entities_boundary(
            mesh, fdim, lambda x: np.isclose(x[1], 0)
        )
        return indices


class RightSurface(F.SurfaceSubdomain):
    def locate_boundary_facet_indices(self, mesh, fdim):
        indices = dolfinx.mesh.locate_entities_boundary(
            mesh, fdim, lambda x: np.isclose(x[0], 1)
        )
        return indices


class LeftSurface(F.SurfaceSubdomain):
    def locate_boundary_facet_indices(self, mesh, fdim):
        indices = dolfinx.mesh.locate_entities_boundary(
            mesh, fdim, lambda x: np.isclose(x[0], 0)
        )
        return indices


left_surface = LeftSurface(id=1)
top_surface = TopSurface(id=2)
bottom_surface = BottomSurface(id=3)
right_surface = RightSurface(id=4)
my_model.subdomains = [
    my_subdomain,
    left_surface,
    top_surface,
    bottom_surface,
    right_surface,
]

mobile_H = F.Species("H")
my_model.species = [mobile_H]

temperature = 500.0
my_model.temperature = temperature

my_model.boundary_conditions = [
    F.DirichletBC(subdomain=top_surface, value=0, species=mobile_H),
    F.DirichletBC(subdomain=bottom_surface, value=0, species=mobile_H),
    F.DirichletBC(subdomain=right_surface, value=0, species=mobile_H),
    F.DirichletBC(subdomain=left_surface, value=0, species=mobile_H),
]

my_model.sources = [
    F.ParticleSource(volume=my_subdomain, species=mobile_H, value=1),
]

my_model.exports = [
    F.XDMFExport("mobile_concentration.xdmf", field=mobile_H),
]

my_model.settings = F.Settings(
    atol=1e-10,
    rtol=1e-10,
    max_iterations=30,
    transient=False,
)

my_model.initialise()


my_model.run()
