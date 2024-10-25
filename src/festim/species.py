from festim.subdomain.volume_subdomain import (
    VolumeSubdomain as _VolumeSubdomain,
)


class Species:
    """
    Hydrogen species class for H transport simulation.

    Args:
        name (str, optional): a name given to the species. Defaults to None.
        mobile (bool, optional): whether the species is mobile or not.
            Defaults to True.
        subdomain (F.VolumeSubdomain, optional): the volume subdomain where the
            species is. Defaults to None.

    Attributes:
        name (str): a name given to the species.
        mobile (bool): whether the species is mobile or not.
        solution (dolfinx.fem.Function): the solution for the current timestep
        prev_solution (dolfinx.fem.Function): the solution for the previous
            timestep
        test_function (ufl.Argument): the testfunction associated with this
            species
        sub_function_space (dolfinx.fem.function.FunctionSpaceBase): the
            subspace of the function space
        collapsed_function_space (dolfinx.fem.function.FunctionSpaceBase): the
            collapsed function space for a species in the function space. In
            case single species case, this is None.
        post_processing_solution (dolfinx.fem.Function): the solution for post
            processing
        concentration (dolfinx.fem.Function): the concentration of the species
        subdomains (F.VolumeSubdomain): the volume subdomains where the species is
        subdomain_to_solution (dict): a dictionary mapping subdomains to solutions
        subdomain_to_prev_solution (dict): a dictionary mapping subdomains to
            previous solutions
        subdomain_to_test_function (dict): a dictionary mapping subdomains to
            test functions
        subdomain_to_post_processing_solution (dict): a dictionary mapping
            subdomains to post processing solutions
        subdomain_to_collapsed_function_space (dict): a dictionary mapping
            subdomains to collapsed function spaces
        subdomain_to_function_space (dict): a dictionary mapping subdomains to
            function spaces

    Usage:
        >>> from festim import Species, HTransportProblem
        >>> species = Species(name="H")
        >>> species.name
        'H'
        >>> my_model = HTransportProblem()
        >>> my_model.species.append(species)

    """

    subdomains: list[_VolumeSubdomain] | _VolumeSubdomain
    subdomain_to_solution: dict
    subdomain_to_prev_solution: dict
    subdomain_to_test_function: dict
    subdomain_to_post_processing_solution: dict
    subdomain_to_collapsed_function_space: dict
    subdomain_to_function_space: dict

    def __init__(self, name: str = None, mobile=True, subdomains=None) -> None:
        self.name = name
        self.mobile = mobile
        self.solution = None
        self.prev_solution = None
        self.test_function = None
        self.sub_function_space = None
        self.post_processing_solution = None
        self.collapsed_function_space = None

        self.subdomains = subdomains
        self.subdomain_to_solution = {}
        self.subdomain_to_prev_solution = {}
        self.subdomain_to_test_function = {}
        self.subdomain_to_post_processing_solution = {}
        self.subdomain_to_collapsed_function_space = {}
        self.subdomain_to_function_space = {}

    def __repr__(self) -> str:
        return f"Species({self.name})"

    def __str__(self) -> str:
        return f"{self.name}"

    @property
    def concentration(self):
        return self.solution

    @property
    def legacy(self) -> bool:
        """
        Check if we are using FESTIM 1.0 implementation or FESTIM 2.0
        """
        if not self.subdomain_to_solution:
            return True
        else:
            return False


class ImplicitSpecies:
    """Implicit species class for H transport simulation.
    c = n - others

    Args:
        n (float): the total concentration of the species
        others (list[Species]): the list of species from which the implicit
            species concentration is computed (c = n - others)
        name (str, optional): a name given to the species. Defaults to None.

    Attributes:
        name (str): a name given to the species.
        n (float): the total concentration of the species
        others (list[Species]): the list of species from which the implicit
            species concentration is computed (c = n - others)
        concentration (form): the concentration of the species

    """

    def __init__(
        self,
        n: float,
        others: list[Species] = None,
        name: str = None,
    ) -> None:
        self.name = name
        self.n = n
        self.others = others

    def __repr__(self) -> str:
        return f"ImplicitSpecies({self.name}, {self.n}, {self.others})"

    def __str__(self) -> str:
        return f"{self.name}"

    @property
    def concentration(self):
        if len(self.others) > 0:
            for other in self.others:
                if other.solution is None:
                    raise ValueError(
                        f"Cannot compute concentration of {
                            self.name} because {other.name} has no solution"
                    )
        return self.n - sum([other.solution for other in self.others])


def find_species_from_name(name: str, species: list):
    """Returns the correct species object from a list of species
    based on a string

    Args:
        name (str): the name of the species
        species (list): the list of species

    Returns:
        species (festim.Species): the species object with the correct name

    Raises:
        ValueError: if the species name is not found in the list of species

    """
    for spe in species:
        if spe.name == name:
            return spe
    raise ValueError(f"Species {name} not found in list of species")