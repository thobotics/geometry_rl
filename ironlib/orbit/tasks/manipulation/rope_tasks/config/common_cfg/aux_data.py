from __future__ import annotations
from dataclasses import dataclass


@dataclass
class AuxiliaryData:
    """Auxiliary additional data.
    These data will be computed once in the beginning and stored for future use.
    """

    rope_geometry_positions = None
    rope_edges = None
    target_geometry_positions = None


aux_data = AuxiliaryData()
