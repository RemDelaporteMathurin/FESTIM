import festim as F
from ufl import exp
import math
import numpy as np


class CavityReaction(F.Reaction):
    def __init__(
        self,
        N_t_eff: F.Species,
        mobile: F.Species,
        product: F.Species,
        R: float,
        lambda_: float,
        a_m: float,
        nu_bs: float,
        E_BS: float,
        nu_sb: float,
        E_SB: float,
        D_0: float,
        E_D: float,
        volume: F.VolumeSubdomain1D,
    ) -> None:
        self.N_t_eff = N_t_eff
        self.mobile = mobile
        self.R = R
        self.lambda_ = lambda_
        self.a_m = a_m
        self.E_BS = E_BS
        self.E_SB = E_SB
        self.E_D = E_D
        self.D_0 = D_0
        self.nu_bs = nu_bs
        self.nu_sb = nu_sb

        self.volume = volume

        super().__init__(
            reactant1=mobile,
            reactant2=F.ImplicitSpecies(n=0),
            product=product,
            k_0=0,
            E_k=0,
            p_0=0,
            E_p=0,
            volume=volume,
        )

    @property
    def N_m(self):
        return self.a_m * 4 * math.pi * self.R**2

    def omega(self, c_t, temperature):
        """Eq. 20

        Args:
            c_t (_type_): _description_
            temperature (_type_): _description_

        Returns:
            _type_: _description_
        """
        return 1 + self.nu_bs * self.lambda_ * self.R / self.D_0 * (
            1 - c_t / self.N_t_eff
        ) * exp((self.E_D - self.E_BS) / (F.k_B * temperature))

    def reaction_term(self, temperature):
        c_t = self.product.concentration
        mobile = self.mobile.concentration

        omega = self.omega(c_t=c_t, temperature=temperature)

        trapping_rate = (
            4
            * math.pi
            * self.R**2
            * self.lambda_
            / self.N_m
            * self.nu_bs
            / omega
            * exp(-self.E_BS / (F.k_B * temperature))
            * (self.N_t_eff - c_t)
            * mobile
        )
        detrapping_rate = (
            self.nu_sb / omega * exp(-self.E_SB / (F.k_B * temperature)) * c_t
        )
        return trapping_rate - detrapping_rate


my_model = F.HydrogenTransportProblem()

# -------- Mesh --------- #

L = 1e-6
vertices = np.linspace(0, L, num=1000)
my_model.mesh = F.Mesh1D(vertices)


# -------- Materials and subdomains --------- #

w_atom_density = 6.306e28  # atom/m3

tungsten = F.Material(D_0=1.5e-7, E_D=0.265, name="tungsten")

my_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, L], material=tungsten)
left_surface = F.SurfaceSubdomain1D(id=1, x=0)
right_surface = F.SurfaceSubdomain1D(id=2, x=L)

my_model.subdomains = [
    my_subdomain,
    left_surface,
    right_surface,
]

# -------- Hydrogen species and reactions --------- #

mobile_H = F.Species("H")
trapped_H = F.Species("trapped_H1", mobile=False)

N_t_eff = 1e-10 * w_atom_density  # m-3
lambda_val = 1.12e-10  # m

my_model.species = [mobile_H, trapped_H]

my_model.reactions = [
    CavityReaction(
        N_t_eff=N_t_eff,
        mobile=mobile_H,
        product=trapped_H,
        R=1e-8,  # m,
        lambda_=lambda_val,
        a_m=6 * w_atom_density * lambda_val,  # at / W * W/m3 * m = at/m2,
        nu_bs=tungsten.D_0 / lambda_val**2,
        E_BS=tungsten.E_D,
        nu_sb=1e13,
        E_SB=1.5,
        E_D=tungsten.E_D,
        D_0=tungsten.D_0,
        volume=my_subdomain,
    )
]

my_model.initial_conditions = [F.InitialCondition(value=N_t_eff, species=trapped_H)]

# -------- Temperature --------- #

implantation_temp = 400
temperature_ramp = 0.1  # K/s
final_temp = 750


def temp_function(t):
    return implantation_temp + temperature_ramp * t


my_model.temperature = temp_function

# -------- Boundary conditions --------- #

my_model.boundary_conditions = [
    F.DirichletBC(subdomain=left_surface, value=0, species=mobile_H),
    F.DirichletBC(subdomain=right_surface, value=0, species=mobile_H),
]

# -------- Exports --------- #

left_flux = F.SurfaceFlux(field=mobile_H, surface=left_surface)
right_flux = F.SurfaceFlux(field=mobile_H, surface=right_surface)

my_model.exports = [
    F.XDMFExport("cavity/mobile_concentration.xdmf", field=mobile_H),
    F.XDMFExport("cavity/trapped_concentration.xdmf", field=trapped_H),
    left_flux,
    right_flux,
]

# -------- Settings --------- #

my_model.settings = F.Settings(
    atol=1e-15,
    rtol=1e-15,
    max_iterations=30,
    final_time=(final_temp - implantation_temp) / temperature_ramp,
)

my_model.settings.stepsize = F.Stepsize(initial_value=2)

# -------- Run --------- #

my_model.initialise()

my_model.run()

# -------- Save results --------- #

np.savetxt(
    "outgassing_flux_tds.txt",
    np.array(left_flux.data) + np.array(right_flux.data),
)
np.savetxt("times_tds.txt", np.array(left_flux.t))

import matplotlib.pyplot as plt

plt.plot(
    temp_function(np.array(left_flux.t)),
    np.array(left_flux.data) + np.array(right_flux.data),
)
plt.xlabel("T (K)")
plt.ylabel("Outgassing flux (m-2s-1)")
plt.show()
