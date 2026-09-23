"""
STAGE 7: Model 3 -- the final, physically-typed equivariant architecture.

    H&E image (3ch RGB, trivial repr)
        |
    rotation-equivariant encoder (p8m / D8, regular-representation blocks)
        |
    shared equivariant representation (block2 output, FieldType with
    c2 copies of the D8 regular representation)
        |-- classification/pathology head  (GroupPooling -> GAP -> Linear)
        `-- typed rank-2 Q-tensor head     (R2Conv -> FieldType([D8.irrep(1,2)]))
                |
            q1, q2 (dense, per spatial location)
                |
            S = sqrt(q1^2+q2^2), theta = 0.5*atan2(q2,q1)

Model 2 (code/model.py::P4MOrientationNet, the historical post-hoc 8-channel
formulation) is UNCHANGED and untouched -- retained only as the explicitly
labeled comparison model, not used as a base for Model 3.

REPRESENTATION-THEORETIC GUARANTEE: the Q head's output FieldType is
group.irrep(1,2) of D8 (escnn: gspaces.flipRot2dOnR2(N=8)), the SAME
representation established and verified in Phase 1R and the Phase 2
target-generation math (docs_v2/phase2_target_math.md): a genuine 2D
irrep whose rotation-by-alpha matrix is exactly R(2*alpha), verified again
numerically in tests_v2/test_model3.py rather than assumed. This is a
discrete-group (D8) guarantee, not a continuous-SO(2) one -- the model is
only guaranteed exactly equivariant under the 16 elements of D8 (8
rotations x 2 reflections), consistent with Phase 1R's finding that
continuous-angle equivariance is only approximate for a discrete-group
network (see docs_v2/ and results_v2/synthetic_replication/ for that
calibration).

Model 3 is NOT trained in Stage 7. This file is implementation +
architectural verification only.
"""
import numpy as np
import torch
import torch.nn as nn
import escnn.nn as enn
from escnn import gspaces


class Model3TypedEquivariant(nn.Module):
    def __init__(self, n_classes=2, c1=8, c2=8, in_channels=3, group_N=8):
        super().__init__()
        assert group_N == 8, "Model 3 uses the established D8 group only (see Stage 7 instruction #5); do not switch to D4 or a continuous group here"
        self.gspace = gspaces.flipRot2dOnR2(N=group_N)
        self.group = self.gspace.fibergroup
        assert self.group.order() == 16, f"expected D8 (order 16), got order {self.group.order()}"

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
        self.shared_type = block2_type  # the "shared equivariant representation"

        # --- classification / pathology head (secondary) ---
        self.gpool = enn.GroupPooling(block2_type)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.gpool.out_type.size, n_classes)

        # --- typed rank-2 Q-tensor head (PRIMARY scientific output) ---
        self.nematic_irrep = self.group.irrep(1, 2)
        assert self.nematic_irrep.size == 2, (
            f"D8.irrep(1,2) must be the 2D rank-2 representation; got size {self.nematic_irrep.size}. "
            f"This would be a genuine representation-theoretic problem -- STOP, do not proceed silently."
        )
        q_out_type = enn.FieldType(self.gspace, [self.nematic_irrep])
        self.q_head = enn.R2Conv(block2_type, q_out_type, kernel_size=3, padding=1, bias=False)
        self.q_out_type = q_out_type

    def forward(self, x, return_dense_q=True):
        """
        x: (B, 3, H, W) raw H&E tensor. NO mask, target, or target-derived
        feature is ever accepted as an input here -- see
        tests_v2/test_no_leakage.py's harness/discipline, which this
        signature also satisfies (single tensor argument, image only).

        Returns:
          logits: (B, n_classes) -- classification head output
          q_dense: (B, 2, H2, W2) -- (q1, q2) at the shared representation's
                    spatial resolution (H2 = H/2, W2 = W/2, from the one
                    stride-2 pool in block1). Dense, NOT globally pooled --
                    per Stage 7 architecture, the Q head produces a
                    spatially-resolved field to be compared against the
                    (resolution-matched, see model3_loss.py) Stage 4
                    anatomical target, not a single pooled vector.
        """
        xg = enn.GeometricTensor(x, self.input_type)
        f1 = self.block1(xg)
        f1 = self.pool1(f1)
        shared = self.block2(f1)  # shared equivariant representation

        pooled = self.gpool(shared)
        feat = self.global_avg_pool(pooled.tensor).flatten(1)
        logits = self.classifier(feat)

        q_field = self.q_head(shared)
        q_dense = q_field.tensor  # (B, 2, H2, W2)

        if return_dense_q:
            return logits, q_dense
        return logits

    def orientation_angles_note(self):
        """
        Documentation-only helper (mirrors nematic_model.py's
        orientation_angles, retained for consistency): this model's Q head
        does NOT use the post-hoc regular-representation-channel-to-angle
        construction at all (that is Model 2's approach). The Q head's
        output channels ARE q1, q2 directly, typed by construction via the
        D8.irrep(1,2) FieldType -- there is no channel-to-angle mapping to
        document or permute here, consistent with Stage 7 instruction #2/§14
        and the master prompt's section 8 (permutation-null test applies
        only to Model 2).
        """
        return "Model 3's Q head has no arbitrary channel-to-angle mapping by construction."
