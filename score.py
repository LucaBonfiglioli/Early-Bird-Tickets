import torch

# n - Number of parameters
# m - Number of snapshots
# w - [n, m] Value of weights in each snapshot
# i - [m] Iterations of each snapshot
# s - [n] Score of each parameter


def large_final(w: torch.Tensor, i: torch.Tensor):
    s = w[:, -1].abs()
    return s


def growing(w: torch.Tensor, i: torch.Tensor):
    s = w[:, -1].abs() - w[:, 0].abs()
    return s


def large_final_growing(w: torch.Tensor, i: torch.Tensor):
    s = 2*w[:, -1].abs() - w[:, 0].abs()
    return s

