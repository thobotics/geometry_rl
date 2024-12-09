from __future__ import annotations
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm

from ... import mdp


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
