"""Advection-diffusion solver written as a convolutional neural network (NN4PDEs).

This module holds the same discretisation as ``Python Code/2D_Advection-Diffusion.py``,
split into small functions so that it can be imported and tested.

The convolution filters are the finite-element stencils for the Laplacian and the two
first derivatives on a uniform grid. Their weights are fixed by the discretisation,
not learned, so the network needs no training.
"""
import torch
from torch import nn

# Finite-element stencils (9-point) on a uniform grid with spacing dx = dy.
DIFF = torch.tensor([[1 / 3, 1 / 3, 1 / 3],
                     [1 / 3, -8 / 3, 1 / 3],
                     [1 / 3, 1 / 3, 1 / 3]])
XADV = torch.tensor([[1 / 12, 0.0, -1 / 12],
                     [1 / 3, 0.0, -1 / 3],
                     [1 / 12, 0.0, -1 / 12]])
YADV = torch.tensor([[-1 / 12, -1 / 3, -1 / 12],
                     [0.0, 0.0, 0.0],
                     [1 / 12, 1 / 3, 1 / 12]])


def _fixed_conv(weights: torch.Tensor) -> nn.Conv2d:
    """A 3x3 convolution whose weights are the given stencil and whose bias is zero."""
    conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
    with torch.no_grad():
        conv.weight.copy_(weights.view(1, 1, 3, 3))
        conv.bias.zero_()
    for p in conv.parameters():
        p.requires_grad_(False)
    return conv


class AdvDiff(nn.Module):
    """One explicit time step of the 2D advection-diffusion equation.

    values_new = values + dt/ml * ( k * L(values)/dx**2 + Dx(values)/dx + Dy(values)/dx )

    where L, Dx and Dy are the stencils above and ml is the lumped mass.
    """

    def __init__(self, k: float = 0.5, dx: float = 1.0, dt: float = 0.01, ml: float = 0.25):
        super().__init__()
        self.k, self.dx, self.dt, self.ml = float(k), float(dx), float(dt), float(ml)
        self.diff = _fixed_conv(DIFF)
        self.xadv = _fixed_conv(XADV)
        self.yadv = _fixed_conv(YADV)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        rhs = (self.k * self.diff(values) / self.dx ** 2
               + self.xadv(values) / self.dx
               + self.yadv(values) / self.dx)
        return values + rhs / self.ml * self.dt


def square_initial_condition(nx: int = 128, ny: int = 128, size: int = 40, offset: int = 43,
                             device="cpu") -> torch.Tensor:
    """A unit square of side ``size`` cells, placed ``offset`` cells from the top-left corner.

    Returns a tensor of shape (1, 1, ny, nx), as the network expects.
    """
    values = torch.zeros((1, 1, ny, nx), device=device)
    values[0, 0, offset:offset + size, offset:offset + size] = 1.0
    return values


def run(model: nn.Module, values: torch.Tensor, ntime: int) -> torch.Tensor:
    """Advance ``values`` by ``ntime`` steps without tracking gradients."""
    with torch.no_grad():
        for _ in range(ntime):
            values = model(values)
    return values


def default_timestep(u: float = 5.0, dx: float = 1.0, cfl: float = 0.05) -> float:
    """The time step used in the original script: dt = cfl * dx / u."""
    return cfl * dx / u
