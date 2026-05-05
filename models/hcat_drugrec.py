"""HCATDrugRec - hierarchical attention model for MIMIC-III drug recommendation.

Three additive logit streams:
    encoder_logits   = head(GRU(visit-aware encoded by intra-visit transformer))
    copy_logits      = recent_mh * w_recent + any_mh * w_any   (silent on cold-start)
    evidence_logits  = Linear(current_visit_codes)             (always fires)
    final_logits     = encoder_logits + copy_logits + evidence_logits

Loss: focal BCE (gamma=2.0) when use_focal=True, else vanilla BCE.

Ablation flags:
    use_focal     - focal loss vs. vanilla BCE
    use_copy      - additive copy stream from drug history
    use_evidence  - additive evidence stream from current-visit codes

The intra-visit transformer + cross-visit GRU + shared dx/proc encoder are
always on - that is the architecture, not a flag.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel

from models.losses import focal_bce_loss


class HCATDrugRec(BaseModel):
    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        intra_heads: int = 2,
        dropout: float = 0.3,
        use_focal: bool = True,
        use_copy: bool = True,
        use_evidence: bool = True,
        focal_gamma: float = 2.0,
        hist_to_label: torch.Tensor | None = None,
    ):
        super().__init__(dataset=dataset)
        d = embedding_dim
        self.embedding_dim = d
        self.use_focal = use_focal
        self.use_copy = use_copy
        self.use_evidence = use_evidence
        self.focal_gamma = focal_gamma

        assert len(self.label_keys) == 1, "Single-label-key task expected"
        self.label_key = self.label_keys[0]
        self.label_size = self.get_output_size()

        proc = dataset.input_processors
        self.cond_vocab_size = proc["conditions"].vocab_size()
        self.proc_vocab_size = proc["procedures"].vocab_size()
        self.drug_vocab_size = proc["drugs_hist"].vocab_size()

        self.cond_emb = nn.Embedding(self.cond_vocab_size, d, padding_idx=0)
        self.proc_emb = nn.Embedding(self.proc_vocab_size, d, padding_idx=0)
        self.drug_emb = nn.Embedding(self.drug_vocab_size, d, padding_idx=0)
        self.type_emb = nn.Embedding(2, d)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        # norm_first=True is more stable for small data
        self.intra = nn.TransformerEncoderLayer(
            d_model=d, nhead=intra_heads, dim_feedforward=4 * d,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )

        self.cross_gru = nn.GRU(input_size=d, hidden_size=d, batch_first=True)

        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Dropout(dropout),
            nn.Linear(d, self.label_size),
        )

        if use_copy:
            assert hist_to_label is not None, "use_copy requires hist_to_label tensor"
            self.register_buffer("hist_to_label", hist_to_label.long())
            self.w_recent = nn.Parameter(torch.zeros(self.label_size))
            self.w_any = nn.Parameter(torch.zeros(self.label_size))

        if use_evidence:
            self.evidence_head = nn.Linear(d, self.label_size, bias=False)
            nn.init.normal_(self.evidence_head.weight, mean=0.0, std=0.01)

    def _intra_visit_encode(self, conditions: torch.Tensor, procedures: torch.Tensor):
        """Encode each visit via the intra-visit transformer; returns [B, V, d]."""
        B, V, C = conditions.shape
        P = procedures.shape[2]
        d = self.embedding_dim

        cond_e = self.cond_emb(conditions) + self.type_emb.weight[0]   # [B, V, C, d]
        proc_e = self.proc_emb(procedures) + self.type_emb.weight[1]   # [B, V, P, d]
        codes = torch.cat([cond_e, proc_e], dim=2)                     # [B, V, C+P, d]

        is_pad = torch.cat([conditions == 0, procedures == 0], dim=2)  # [B, V, C+P]

        T = C + P
        codes = codes.reshape(B * V, T, d)
        is_pad = is_pad.reshape(B * V, T)

        cls = self.cls_token.expand(B * V, -1, -1)
        codes = torch.cat([cls, codes], dim=1)
        cls_pad = torch.zeros(B * V, 1, dtype=torch.bool, device=is_pad.device)
        pad_mask = torch.cat([cls_pad, is_pad], dim=1)

        # Rows where every code is padded would NaN out attention; flip the
        # first non-CLS slot to non-pad for those rows.
        all_pad = pad_mask[:, 1:].all(dim=1)
        if all_pad.any():
            pad_mask = pad_mask.clone()
            pad_mask[all_pad, 1] = False

        out = self.intra(codes, src_key_padding_mask=pad_mask)
        return out[:, 0, :].reshape(B, V, d)

    def _build_multihots(self, drugs_hist: torch.Tensor):
        """Build (recent_mh, any_mh) each [B, label_size]."""
        B, V, D = drugs_hist.shape
        device = drugs_hist.device
        L = self.label_size

        hist_label_ids = self.hist_to_label[drugs_hist.clamp(min=0)]
        valid = (hist_label_ids >= 0) & (drugs_hist != 0)

        any_mh = torch.zeros(B, L, device=device)
        flat_ids = hist_label_ids.clamp(min=0).reshape(B, -1)
        flat_valid = valid.reshape(B, -1).float()
        any_mh.scatter_add_(1, flat_ids, flat_valid)
        any_mh = (any_mh > 0).float()

        per_visit_count = valid.sum(dim=2)
        per_visit_has = (per_visit_count > 0).long()
        v_idx = torch.arange(V, device=device)[None, :].expand(B, V)
        masked = v_idx * per_visit_has + (per_visit_has - 1)
        last_idx = masked.max(dim=1).values

        b_idx = torch.arange(B, device=device)
        gather_idx = last_idx.clamp(min=0)
        recent_label_ids = hist_label_ids[b_idx, gather_idx]
        recent_valid = valid[b_idx, gather_idx]
        no_hist = (last_idx < 0).unsqueeze(1)
        recent_valid = recent_valid & ~no_hist

        recent_mh = torch.zeros(B, L, device=device)
        recent_mh.scatter_add_(1, recent_label_ids.clamp(min=0), recent_valid.float())
        recent_mh = (recent_mh > 0).float()

        return recent_mh, any_mh

    def forward(self, **kwargs):
        device = self.device
        conditions = kwargs["conditions"].to(device).long()    # [B, V, C]
        procedures = kwargs["procedures"].to(device).long()    # [B, V, P]
        drugs_hist = kwargs["drugs_hist"].to(device).long()    # [B, V, D]
        y_true = kwargs[self.label_key].to(device).float()     # [B, L]

        B = conditions.shape[0]

        v_codes = self._intra_visit_encode(conditions, procedures)
        v_drugs = self.drug_emb(drugs_hist).sum(dim=2)
        per_visit = v_codes + v_drugs

        # Gather GRU output at each patient's last *real* visit. PyHealth pads
        # V to the batch max, so h_n would come from padded steps for shorter
        # patients.
        gru_out, _ = self.cross_gru(per_visit)
        visit_has_codes = (conditions != 0).any(dim=2)
        valid_lens = visit_has_codes.sum(dim=1).clamp(min=1)
        b_idx = torch.arange(B, device=device)
        encoder_repr = gru_out[b_idx, valid_lens - 1]

        logits = self.head(encoder_repr)

        if self.use_copy:
            recent_mh, any_mh = self._build_multihots(drugs_hist)
            logits = logits + recent_mh * self.w_recent + any_mh * self.w_any

        if self.use_evidence:
            current_codes = v_codes[b_idx, valid_lens - 1]
            logits = logits + self.evidence_head(current_codes)

        if self.use_focal:
            loss = focal_bce_loss(logits, y_true, gamma=self.focal_gamma)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, y_true)

        y_prob = torch.sigmoid(logits)
        return {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}


def build_hist_to_label_map(samples) -> torch.Tensor:
    """Map drugs_hist vocab indices to label vocab indices (or -1 if absent)."""
    hist_vocab = samples.input_processors["drugs_hist"].code_vocab
    label_vocab = samples.output_processors["drugs"].label_vocab
    hist_size = samples.input_processors["drugs_hist"].vocab_size()
    out = torch.full((hist_size,), -1, dtype=torch.long)
    for code, hist_idx in hist_vocab.items():
        if code in label_vocab:
            out[hist_idx] = label_vocab[code]
    return out
