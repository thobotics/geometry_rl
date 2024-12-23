from torch import nn

class BaseCritic(nn.Module):
    def __init__(self, vf):
        super().__init__()

        self._network1 = vf

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
        return True
