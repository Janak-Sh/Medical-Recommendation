"""HCATBinary - intra-visit attention model for binary outcome prediction.

Used for both mortality and readmission, since each sample is one visit and
the architecture is otherwise identical: predict a single binary label from
the visit's clinical codes plus structured admin features.

Two parallel paths produce d-dim representations and are summed before the head:
    code_repr = intra-visit transformer + [CLS] pooling over [dx, proc, drug]
    tab_repr  = MLP over [age_z, los_z, n_prior_z, admit_emb, disch_emb]

Either path can be ablated via use_codes / use_admin (at least one must be on).

`init_bias_from_marginal` seeds the head bias with log-odds of the train-set
positive rate, which materially helps optimization on imbalanced binary tasks.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel

from data.admin_features import N_ADMIN_NUMERIC
from models.losses import focal_bce_loss


class HCATBinary(BaseModel):
    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        intra_heads: int = 2,
        dropout: float = 0.3,
        use_codes: bool = True,
        use_admin: bool = True,
        use_focal: bool = True,
        focal_gamma: float = 2.0,
        admit_emb_dim: int = 16,
        disch_emb_dim: int = 16,
    ):
        super().__init__(dataset=dataset)
        d = embedding_dim
        self.embedding_dim = d
        self.use_codes = use_codes
        self.use_admin = use_admin
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma

        assert use_codes or use_admin, "At least one of {codes, admin} must be enabled"
        assert len(self.label_keys) == 1
        self.label_key = self.label_keys[0]

        proc = dataset.input_processors

        def vsize(key):
            return len(proc[key].code_vocab)

        if use_codes:
            self.cond_emb = nn.Embedding(vsize("conditions"), d, padding_idx=0)
            self.proc_emb = nn.Embedding(vsize("procedures"), d, padding_idx=0)
            self.drug_emb = nn.Embedding(vsize("drugs"),      d, padding_idx=0)
            self.type_emb = nn.Embedding(3, d)  # dx=0, proc=1, drug=2
            self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            self.intra = nn.TransformerEncoderLayer(
                d_model=d, nhead=intra_heads, dim_feedforward=4 * d,
                dropout=dropout, batch_first=True, norm_first=True,
                activation="gelu",
            )

        if use_admin:
            assert "admit_type" in proc and "discharge_location" in proc and "admin_numeric" in proc, (
                "use_admin=True requires the *WithAdminMIMIC3 task variant "
                "(which exposes admit_type, discharge_location, admin_numeric)."
            )
            self.admit_emb = nn.Embedding(vsize("admit_type"),         admit_emb_dim, padding_idx=0)
            self.disch_emb = nn.Embedding(vsize("discharge_location"), disch_emb_dim, padding_idx=0)
            admin_in = N_ADMIN_NUMERIC + admit_emb_dim + disch_emb_dim
            self.admin_mlp = nn.Sequential(
                nn.Linear(admin_in, d),
                nn.GELU(),
                nn.LayerNorm(d),
                nn.Dropout(dropout),
                nn.Linear(d, d),
            )

        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Dropout(dropout),
            nn.Linear(d, 1),
        )

    def init_bias_from_marginal(self, positive_rate: float) -> None:
        """Initialize the head bias to log-odds of the train-set prior."""
        p = float(min(max(positive_rate, 1e-6), 1.0 - 1e-6))
        self.head[-1].bias.data.fill_(math.log(p / (1 - p)))

    def _encode_codes(self, cond, proc, drug):
        cond_e = self.cond_emb(cond) + self.type_emb.weight[0]
        proc_e = self.proc_emb(proc) + self.type_emb.weight[1]
        drug_e = self.drug_emb(drug) + self.type_emb.weight[2]
        codes  = torch.cat([cond_e, proc_e, drug_e], dim=1)            # [B, T, d]

        is_pad = torch.cat([cond == 0, proc == 0, drug == 0], dim=1)   # [B, T]

        B = codes.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        codes = torch.cat([cls, codes], dim=1)
        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=cond.device)
        pad_mask = torch.cat([cls_pad, is_pad], dim=1)
        all_pad = pad_mask[:, 1:].all(dim=1)
        if all_pad.any():
            pad_mask = pad_mask.clone()
            pad_mask[all_pad, 1] = False
        out = self.intra(codes, src_key_padding_mask=pad_mask)
        return out[:, 0, :]

    def _encode_admin(self, numeric, admit, disch):
        admit_e = self.admit_emb(admit).squeeze(1)
        disch_e = self.disch_emb(disch).squeeze(1)
        admin_concat = torch.cat([numeric, admit_e, disch_e], dim=1)
        return self.admin_mlp(admin_concat)

    def forward(self, **kwargs):
        device = self.device
        y_true = kwargs[self.label_key].to(device).float().view(-1)

        patient_repr = None

        if self.use_codes:
            cond = kwargs["conditions"].to(device).long()
            proc = kwargs["procedures"].to(device).long()
            drug = kwargs["drugs"].to(device).long()
            patient_repr = self._encode_codes(cond, proc, drug)

        if self.use_admin:
            numeric = kwargs["admin_numeric"].to(device).float()
            admit   = kwargs["admit_type"].to(device).long()
            disch   = kwargs["discharge_location"].to(device).long()
            tab_repr = self._encode_admin(numeric, admit, disch)
            patient_repr = tab_repr if patient_repr is None else patient_repr + tab_repr

        logit = self.head(patient_repr).squeeze(-1)

        if self.use_focal:
            loss = focal_bce_loss(logit, y_true, gamma=self.focal_gamma)
        else:
            loss = F.binary_cross_entropy_with_logits(logit, y_true)

        y_prob = torch.sigmoid(logit)
        return {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logit}
