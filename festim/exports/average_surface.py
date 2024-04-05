import festim as F
import numpy as np


class AverageSurface(F.SurfaceQuantity):
    """Computes the average value of a field on a given surface

    Args:
        field (festim.Species): species for which the average surface is computed
        surface (festim.SurfaceSubdomain): surface subdomain
        filename (str, optional): name of the file to which the average surface is exported

    Attributes:
        see `festim.SurfaceQuantity`
    """

    def __init__(
        self,
        field: F.Species,
        surface: F.SurfaceSubdomain,
        filename: str = None,
    ) -> None:
        super().__init__(field=field, surface=surface, filename=filename)
    
    @property
    def title(self):
        return f"Average {self.field.name} surface {self.surface.id}"

    def compute(self):
        """
        Computes the average value of the field on the defined surface
        subdomain, and appends it to the data list
        """
        self.value = np.mean(self.field.solution.x.array[self.surface.indices])
        self.data.append(self.value)