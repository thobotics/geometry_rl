from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AuxiliaryData:
    """Auxiliary additional data.
    These data will be computed once in the beginning and stored for future use.
    """

    object_geometry_positions = None
    object_geometry_edges = None
    num_points = None
    num_edges = None


aux_data = AuxiliaryData()
