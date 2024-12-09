from __future__ import annotations
from dataclasses import dataclass


@dataclass
class AuxiliaryData:
    """Auxiliary additional data.
    These data will be computed once in the beginning and stored for future use.
    """

    cloth_geometry_positions = None
    cloth_triangles = None
    cloth_edges = None
    holes_boundary_nodes_indices = None
    target_hook_geometry_positions = None


aux_data = AuxiliaryData()
