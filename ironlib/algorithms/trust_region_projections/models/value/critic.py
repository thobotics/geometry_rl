import copy
from typing import Union, List

import torch as ch
from torch import nn

from .qf_net import QFNet
from .vf_net import VFNet
from ...utils.network_utils import polyak_update


class Critic(nn.Module):
    def __init__(self, critics: List[Union[QFNet, VFNet]], use_target_net: bool = True):
        super().__init__()

        self._is_vf = isinstance(critics[0], VFNet)
        self.critics = critics

        self.target_critics = None
        if use_target_net:
            self.target_critics = copy.deepcopy(self.critics)
            for net in self.target_critics:
                for p in net.parameters():
                    p.requires_grad = False

    def forward(self, x, *args, **kwargs):
        return [critic(x, *args, **kwargs) for critic in self.critics]

    def target(self, x, *args, **kwargs):
        if self.target_critics is not None:
            return [critic(x, *args, **kwargs) for critic in self.target_critics]
        else:
            raise ValueError("Currently no target networks in use.")

    def update_target_net(self, polyak=1.0):
        """
        Polyak update of the target network(s).
        A polyak weight of 1 -> completely copy weights from qf
        Args:
            polyak: polyak update weight

        Returns:

        """
        for source, target in zip(self.critics, self.target_critics):
            polyak_update(source, target, polyak)


class BaseCritic(nn.Module):
    def __init__(self, qf: Union[QFNet, VFNet]):
        super().__init__()

        self._network1 = qf
        self._is_vf = isinstance(self._network1, VFNet)

    def forward(self, x, *args, **kwargs):
        return self._network1(x, *args, **kwargs)

    def q1(self, x, *args, **kwargs):
        return self._network1(x, *args, **kwargs)

    def target(self, x, *args, **kwargs):
        raise ValueError("BaseCritic does not have target networks.")

    def update_target_net(self, polyak=1.0):
        """
        Placeholder method in case target networks are used by any other class
        Args:
            polyak:

        Returns:

        """
        pass

    @property
    def is_vf(self):
        return self._is_vf


class TargetCritic(BaseCritic):
    def __init__(self, qf: Union[QFNet, VFNet]):
        super().__init__(qf)

        self._target_network1 = copy.deepcopy(self._network1)
        for p in self._target_network1.parameters():
            p.requires_grad = False

    def target(self, x, *args, **kwargs):
        return self._target_network1(x, *args, **kwargs)

    def update_target_net(self, polyak=1.0):
        """
        Polyak update of the target network(s).
        A polyak weight of 1 -> completely copy weights from qf
        Args:
            polyak: polyak update weight

        Returns:

        """
        polyak_update(self._network1, self._target_network1, polyak)


class DoubleCritic(TargetCritic):
    def __init__(
        self, critic_network1: Union[QFNet, VFNet], critic_network2: Union[QFNet, VFNet]
    ):
        super().__init__(critic_network1)

        self._network2 = critic_network2
        self._target_network2 = copy.deepcopy(self._network2)
        for p in self._target_network2.parameters():
            p.requires_grad = False

    def forward(self, x, *args, **kwargs):
        return ch.min(
            super(DoubleCritic, self).forward(x, *args, **kwargs),
            self._network2(x, *args, **kwargs),
        )

    def q2(self, x, *args, **kwargs):
        return self._network2(x, *args, **kwargs)

    def target(self, x, *args, **kwargs):
        return ch.min(
            super(DoubleCritic, self).target(x, *args, **kwargs),
            self._target_network2(x, *args, **kwargs),
        )

    def update_target_net(self, polyak=1.0):
        super(DoubleCritic, self).update_target_net(polyak)
        polyak_update(self._network2, self._target_network2, polyak)


class DistributionalCritic(TargetCritic):
    def __init__(self, qf: Union[QFNet, VFNet]):
        super().__init__(qf)

        self._target_network1 = copy.deepcopy(self._network1)
        for p in self._target_network1.parameters():
            p.requires_grad = False

    def target(self, x, *args, **kwargs):
        return self._target_network1(x, *args, **kwargs)

    def update_target_net(self, polyak=1.0):
        """
        Polyak update of the target network(s).
        A polyak weight of 1 -> completely copy weights from qf
        Args:
            polyak: polyak update weight

        Returns:

        """
        polyak_update(self._network1, self._target_network1, polyak)

    @property
    def n_atoms(self):
        return self._network1.n_atoms

    @property
    def vmin(self):
        return self._target_network1.vmin

    @property
    def vmax(self):
        return self._target_network1.vmax

    def get_value(self, value_dist):
        return ch.sum(
            self._target_network1.atoms * ch.softmax(value_dist, dim=-1), dim=-1
        )
