import festim as F
import numpy as np
import fenics as f
from analytical_enclosure import analytical_expression_fractional_release

encl_vol = 5.20e-11  # m3
encl_surf = 2.16e-6  # m2
l = 3.3e-5  # m
R = 8.314
avogadro = 6.022e23  # mol-1
temperature = 2373  # K
initial_pressure = 1e6  # Pa
solubility = 7.244e22 / temperature  # H/m3/Pa
diffusivity = 2.6237e-11  # m2/s


def henrys_law(T, S_0, E_S, pressure):
    S = S_0 * f.exp(-E_S / F.k_B / T)
    return S * pressure


class PressureExport(F.DerivedQuantity):
    def __init__(self, **kwargs):
        super().__init__(field="solute", **kwargs)
        self.title = "enclosure_pressure"
        self.data = []

    def compute(self):
        return float(left_bc.pressure)


class CustomHenrysBC(F.HenrysBC):
    def create_expression(self, T):
        value_BC = F.BoundaryConditionExpression(
            T,
            henrys_law,
            S_0=self.H_0,
            E_S=self.E_H,
            pressure=self.pressure,
        )
        self.expression = value_BC
        self.sub_expressions = [self.pressure]


class CustomSimulation(F.Simulation):
    def iterate(self):
        super().iterate()
        # Update pressure based on flux
        left_flux_val = left_flux.compute()
        old_pressure = float(left_bc.pressure)
        new_pressure = (
            old_pressure
            - (left_flux_val * encl_surf / encl_vol * R * self.T.T(0) / avogadro)
            * self.dt.value
        )
        left_bc.pressure.assign(new_pressure)


my_model = CustomSimulation()

vertices = np.linspace(0, l, 10)

my_model.mesh = F.MeshFromVertices(vertices)

my_model.materials = F.Material(
    id=1,
    D_0=diffusivity,
    E_D=0,
)

left_bc = CustomHenrysBC(
    surfaces=1, H_0=solubility, E_H=0, pressure=f.Constant(initial_pressure)
)

my_model.boundary_conditions = [
    left_bc,
    F.DirichletBC(surfaces=2, value=0, field="solute"),
]

my_model.T = F.Temperature(temperature)

my_model.settings = F.Settings(
    absolute_tolerance=1e10,
    relative_tolerance=1e-10,
    final_time=200,
)

left_flux = F.HydrogenFlux(surface=1)
right_flux = F.HydrogenFlux(surface=2)
pressure_export = PressureExport()
derived_quantities = F.DerivedQuantities([left_flux, right_flux, pressure_export])

my_model.exports = [
    F.XDMFExport("solute", filename="enclosure/mobile.xdmf", checkpoint=False),
    derived_quantities,
]

my_model.dt = F.Stepsize(initial_value=0.1)


my_model.initialise()
my_model.run()
print(f"final pressure is {float(left_bc.pressure)}")

t = derived_quantities.t
pressures = np.array(pressure_export.data)
fractional_release = 1 - pressures / initial_pressure
right_flux = np.abs(right_flux.data)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(t, fractional_release, linestyle="--")

times = np.linspace(0, my_model.settings.final_time, 1000)
analytical = analytical_expression_fractional_release(
    t=times,
    P_0=initial_pressure,
    D=my_model.materials.materials[0].D_0,
    S=left_bc.H_0,
    V=encl_vol,
    T=temperature,
    A=encl_surf,
    l=l,
)
plt.plot(times, analytical)
plt.figure()
plt.plot(t, right_flux)
plt.show()
