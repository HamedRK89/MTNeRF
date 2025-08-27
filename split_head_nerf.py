import torch
import torch.nn as nn
import torch.nn.functional as F

class SplitLayerNeRF(nn.Module):
    """
    NeRF with a shared trunk up to `split_at`, then per-head tails + per-head output heads.

    - D, W: depth/width of the MLP over positional-encoded points
    - input_ch:  channels for embedded xyz
    - input_ch_views: channels for embedded viewdirs (0 if not use_viewdirs)
    - output_ch: 4 (RGB+sigma) or 5 if you add extras
    - skips: list of layer indices (global over 0..D-1) that use skip connections to input_pts
    - use_viewdirs: if True, uses the "feature + view MLP → RGB" + alpha head pattern
    - num_heads: number of branches (e.g., 2 -> original & virtual)
    - split_at: the FIRST layer index where we branch into per-head tails (0 < split_at < D)

    Forward expects:
      x: [N_pts, input_ch + (input_ch_views if use_viewdirs else 0)]
      head_idx: [N_pts] LongTensor in {0..num_heads-1} selecting head per sample
    """
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4,
                 skips=[4], use_viewdirs=False, num_heads=2, split_at=4):
        super().__init__()
        assert 0 < split_at < D, f"`split_at` must be in (0, {D}), got {split_at}"
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = set(skips)
        self.use_viewdirs = use_viewdirs
        self.num_heads = num_heads
        self.split_at = split_at

        # -------- Shared trunk: layers 0..split_at-1 --------
        self.shared_layers = nn.ModuleList()
        for i in range(split_at):
            if i == 0:
                self.shared_layers.append(nn.Linear(input_ch, W))
            else:
                if i in self.skips:
                    self.shared_layers.append(nn.Linear(W + input_ch, W))
                else:
                    self.shared_layers.append(nn.Linear(W, W))

        # -------- Per-head tails: layers split_at..D-1 --------
        self.tails = nn.ModuleList()  # list of ModuleList (one per head)
        for _ in range(num_heads):
            tail = nn.ModuleList()
            for g in range(split_at, D):
                if g in self.skips:
                    tail.append(nn.Linear(W + input_ch, W))
                else:
                    tail.append(nn.Linear(W, W))
            self.tails.append(tail)

        # -------- Per-head output heads --------
        if use_viewdirs:
            # Feature → per-head alpha; per-head view MLP → per-head RGB
            self.feature_linear_heads = nn.ModuleList([nn.Linear(W, W) for _ in range(num_heads)])
            self.alpha_linear_heads   = nn.ModuleList([nn.Linear(W, 1) for _ in range(num_heads)])
            self.views_linears_heads  = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2) for _ in range(num_heads)])
            self.rgb_linear_heads     = nn.ModuleList([nn.Linear(W // 2, 3) for _ in range(num_heads)])
        else:
            self.output_linear_heads  = nn.ModuleList([nn.Linear(W, output_ch) for _ in range(num_heads)])

    def _forward_shared(self, input_pts):
        h = input_pts
        for i, layer in enumerate(self.shared_layers):
            g = i  # global index
            if g in self.skips and g > 0:
                h = torch.cat([input_pts, h], dim=-1)
            h = F.relu(layer(h))
        return h  # shared feature of size W

    def _forward_tail_once(self, h_in, input_pts, input_views, head_id):
        """
        Run one head's tail and final outputs.
        Returns: [N_pts, 4] if use_viewdirs else [N_pts, output_ch]
        """
        h = h_in
        tail = self.tails[head_id]
        # Tail layers are global indices split_at..D-1
        for j, layer in enumerate(tail):
            g = self.split_at + j  # global index
            if g in self.skips:
                h = torch.cat([input_pts, h], dim=-1)
            h = F.relu(layer(h))

        if self.use_viewdirs:
            # per-head alpha
            alpha = self.alpha_linear_heads[head_id](h)        # [N,1]
            feat  = self.feature_linear_heads[head_id](h)      # [N,W]
            hv    = torch.cat([feat, input_views], dim=-1)     # [N, W + input_ch_views]
            hv    = F.relu(self.views_linears_heads[head_id](hv))  # [N, W//2]
            rgb   = self.rgb_linear_heads[head_id](hv)         # [N,3]
            out   = torch.cat([rgb, alpha], dim=-1)            # [N,4]
        else:
            out   = self.output_linear_heads[head_id](h)       # [N,C]
        return out

    def forward(self, x, head_idx=None):
        if self.use_viewdirs:
            input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        else:
            input_pts = x
            input_views = None

        if head_idx is None:
            head_idx = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        # Shared trunk
        h_shared = self._forward_shared(input_pts)

        # Compute each head's output for whole batch, then gather
        head_outs = []
        for h in range(self.num_heads):
            out_h = self._forward_tail_once(h_shared, input_pts, input_views, h)  # [N, C]
            head_outs.append(out_h)
        outs = torch.stack(head_outs, dim=1)  # [N, H, C]

        idx = head_idx.view(-1, 1, 1).expand(-1, 1, outs.shape[-1])  # [N,1,C]
        sel = torch.gather(outs, 1, idx).squeeze(1)                  # [N,C]
        return sel
