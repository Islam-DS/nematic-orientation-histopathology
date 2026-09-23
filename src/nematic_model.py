"""
Physically-typed equivariant nematic regressor.

image -> equivariant encoder (regular-rep hidden layers) -> R2Conv into
FieldType(gspace, [group.irrep(1, 2)]) -> equivariant global average pool
-> (q1, q2)

The physical meaning (Q(R_alpha x) = R(2 alpha) Q(x)) is imposed by the
OUTPUT FIELD TYPE itself (group.irrep(1, 2)), not assumed post-hoc from
arbitrary regular-representation channel indices. This is the core design
change from the rejected pilot.

group_N: 8 for p8m (proper 2D frequency-2 irrep, the intended model) or
4 for p4m (architectural sanity comparison only -- see nematic_math.py
docstring: D4's irrep(1,2) is only 1-DIMENSIONAL, so this mode can only
ever predict a degenerate binary q1-like value; it exists here purely to
demonstrate that failure quantitatively, not as a viable model).
"""
import numpy as np
import torch
import torch.nn as nn
import escnn.nn as enn
from escnn import gspaces


class NematicNet(nn.Module):
    def __init__(self, group_N=8, c1=8, c2=8, in_channels=1):
        super().__init__()
        assert group_N in (4, 8), "only p4m (4) and p8m (8) are wired up"
        self.group_N = group_N
        self.gspace = gspaces.flipRot2dOnR2(N=group_N)
        self.group = self.gspace.fibergroup

        in_type = enn.FieldType(self.gspace, in_channels * [self.gspace.trivial_repr])
        self.input_type = in_type

        block1_type = enn.FieldType(self.gspace, c1 * [self.gspace.regular_repr])
        self.block1 = enn.SequentialModule(
            enn.R2Conv(in_type, block1_type, kernel_size=5, padding=2, bias=False),
            enn.InnerBatchNorm(block1_type),
            enn.ReLU(block1_type, inplace=True),
        )
        self.pool1 = enn.PointwiseAvgPoolAntialiased(block1_type, sigma=0.66, stride=2)

        block2_type = enn.FieldType(self.gspace, c2 * [self.gspace.regular_repr])
        self.block2 = enn.SequentialModule(
            enn.R2Conv(block1_type, block2_type, kernel_size=5, padding=2, bias=False),
            enn.InnerBatchNorm(block2_type),
            enn.ReLU(block2_type, inplace=True),
        )

        # Nematic irrep (frequency 2). For group_N=4 this is 1D (degenerate,
        # see module docstring); the Q head and downstream code still work
        # mechanically, they just cannot represent a genuine 2-vector.
        self.nematic_irrep = self.group.irrep(1, 2)
        q_out_type = enn.FieldType(self.gspace, [self.nematic_irrep])
        self.q_head = enn.R2Conv(block2_type, q_out_type, kernel_size=3, padding=1, bias=False)
        self.q_out_type = q_out_type
        self.q_dim = self.nematic_irrep.size  # 2 for p8m, 1 for p4m

        # NOTE: escnn's PointwiseAdaptiveAvgPool asserts the field type
        # supports the 'pointwise' nonlinearity family; group.irrep(1, 2)
        # only supports {'gated', 'norm'} (verified above), so that module
        # rejects this field type. We instead average over space manually
        # on the raw tensor. This is still exactly equivariant: global
        # spatial averaging is a LINEAR operation that (a) commutes with
        # the fixed per-pixel representation matrix applied identically at
        # every location, and (b) is invariant to the spatial permutation
        # a rotation induces (summing over all locations doesn't care
        # about their order). Verified numerically in the equivariance
        # test below (run_synthetic_benchmark.py), not just asserted here.

    def forward(self, x):
        """x: (B, in_channels, H, W) raw tensor. Returns q: (B, q_dim)."""
        x = enn.GeometricTensor(x, self.input_type)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        q_field = self.q_head(x)
        q = q_field.tensor.mean(dim=(2, 3))  # (B, q_dim) -- see NOTE above
        if self.q_dim == 1:
            # p4m degenerate case: pad with zeros so downstream (phi,S)
            # code can treat it uniformly as a 2-vector with q2 forced to 0
            # (i.e. it can only ever express phi in {0, pi/2}).
            q = torch.cat([q, torch.zeros_like(q)], dim=1)
        return q
