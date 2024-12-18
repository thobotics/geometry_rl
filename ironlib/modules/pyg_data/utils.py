import torch


def exponential_weight(range, multiplier=1.0):
    a = (
        -torch.log(torch.tensor(1.0 - 1.0 / range)) * multiplier
    )  # Increase the multiplier (2 in this case) for a steeper curve
    weight = torch.exp(a * torch.arange(0, range))
    normalized_weight = weight / weight[-1]  # Normalize so that weight at step 20 is 1
    return normalized_weight


def noise_like(tensor, std=0.1):
    noise = torch.normal(0.0, std, size=tensor.shape).to(tensor.device)
    return noise


def add_noise(inputs: torch.Tensor, mask: torch.Tensor, noise_std: float = 0.1):
    noise = torch.zeros_like(inputs)
    noise[mask] = noise_like(noise[mask], noise_std)

    return inputs + noise
