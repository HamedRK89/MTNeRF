import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))

class MultiHeadNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3,
                 output_ch=4, skips=[4], use_viewdirs=False, num_heads=2):
        super().__init__()
        self.D, self.W = D, W
        self.input_ch, self.input_ch_views = input_ch, input_ch_views
        self.skips, self.use_viewdirs = skips, use_viewdirs
        self.num_heads = num_heads

        # Shared trunk over 3D points
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] +
            [nn.Linear(W, W) if i not in skips else nn.Linear(W + input_ch, W) for i in range(D-1)]
        )

        if use_viewdirs:
            # Shared 'views' trunk
            self.feature_linear = nn.Linear(W, W)
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

            # Branching heads
            self.alpha_heads = nn.ModuleList([nn.Linear(W, 1) for _ in range(num_heads)])
            self.rgb_heads   = nn.ModuleList([nn.Linear(W // 2, 3) for _ in range(num_heads)])
        else:
            # Single trunk + branching heads on full output
            self.output_heads = nn.ModuleList([nn.Linear(W, output_ch) for _ in range(num_heads)])

    def forward(self, x, head_idx=None):
        """
        x: [N, input_ch + input_ch_views] (if use_viewdirs) or [N, input_ch]
        head_idx: [N] int {0,1} indicating which head to use per sample
        """
        if head_idx is None:
            # default to head 0 (original)
            head_idx = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        if self.use_viewdirs:
            input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        else:
            input_pts, input_views = x, None

        # Shared point MLP
        h = input_pts
        for i, layer in enumerate(self.pts_linears):
            h = F.relu(layer(h))
            if i in self.skips:
                h = torch.cat([input_pts, h], dim=-1)

        if self.use_viewdirs:
            # Shared intermediates
            alpha_shared = h
            feat = self.feature_linear(h)
            hdir = torch.cat([feat, input_views], dim=-1)
            for layer in self.views_linears:
                hdir = F.relu(layer(hdir))

            # Compute all heads, then gather the right one per sample
            alpha_all = torch.stack([head(alpha_shared) for head in self.alpha_heads], dim=1)  # [N,H,1]
            rgb_all   = torch.stack([head(hdir)        for head in self.rgb_heads],   dim=1)  # [N,H,3]

            idx_a = head_idx.view(-1,1,1).expand(-1,1,1)
            idx_r = head_idx.view(-1,1,1).expand(-1,1,3)

            alpha = torch.gather(alpha_all, 1, idx_a).squeeze(1)  # [N,1]
            rgb   = torch.gather(rgb_all,   1, idx_r).squeeze(1)  # [N,3]
            return torch.cat([rgb, alpha], dim=-1)
        else:
            # Compute all heads and pick
            outs_all = torch.stack([head(h) for head in self.output_heads], dim=1)  # [N,H,output_ch]
            idx = head_idx.view(-1,1,1).expand(-1,1,outs_all.shape[-1])
            outs = torch.gather(outs_all, 1, idx).squeeze(1)  # [N,output_ch]
            return outs
        
class SplitLayerNeRF(nn.Module):
    """
    NeRF with shared 8-layer trunk (pts_linears) and configurable branching point:
      - branch_from='heads'   : share trunk + feature_linear + views_linears; split only final heads
      - branch_from='views'   : share trunk + feature_linear; split views_linears + rgb head (alpha head can be shared or per-head)
      - branch_from='feature' : share trunk; split feature_linear + views_linears + rgb head (alpha head can be shared or per-head)

    If use_viewdirs=False, views path is ignored; α head still used.
    """
    def __init__(self, D=8, W=256,
                 input_ch=63,            # xyz embed size
                 input_ch_views=27,      # dir embed size
                 output_ch=4,            # usually 4 or 5 (we'll pad to 5 if needed)
                 skips=[4],
                 use_viewdirs=True,
                 num_heads=2,
                 branch_from='heads',    # 'heads' | 'views' | 'feature'
                 share_alpha=False):     # if True, α head shared even when branching earlier
        super().__init__()
        assert D >= 1
        assert branch_from in ('heads','views1','views','feature')

        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = set(skips or [])
        self.use_viewdirs = use_viewdirs
        self.output_ch = output_ch
        self.num_heads = num_heads
        self.branch_from = branch_from
        self.share_alpha = share_alpha

        # -------------------------
        # Shared 8-layer xyz trunk
        # -------------------------
        self.pts_linears = nn.ModuleList()
        self.pts_linears.append(nn.Linear(input_ch, W))
        for i in range(1, D):
            in_ch = W + (input_ch if i in self.skips else 0)
            self.pts_linears.append(nn.Linear(in_ch, W))

        # -------------------------
        # Alpha (σ) head: shared or per-head
        # (depends only on trunk features)
        # -------------------------
        if share_alpha:
            self.alpha_shared = nn.Linear(W, 1)
            self.alpha_heads = None
        else:
            self.alpha_shared = None
            self.alpha_heads = nn.ModuleList([nn.Linear(W, 1) for _ in range(num_heads)])

        # -------------------------
        # feature_linear: shared or per-head
        # -------------------------
        if branch_from == 'feature':
            self.feature_shared = None
            self.feature_heads = nn.ModuleList([nn.Linear(W, W) for _ in range(num_heads)])
        else:
            self.feature_shared = nn.Linear(W, W)
            self.feature_heads = None


        # -------------------------
        # views_linears (+ rgb head): shared or per-head
        # Follow the common NeRF setting: one view MLP to W//2 then rgb head.
        # -------------------------

        # self.view_hidden = W // 2 if self.use_viewdirs else 0

        # def make_view_block():
        #     # single layer: concat(feature, dir_embed) -> W//2 (or W)
        #     return nn.ModuleList([nn.Linear(W + self.input_ch_views,self.view_hidden)])

        # if self.use_viewdirs:
        
        #     if branch_from in ('views','feature'):
        #         self.views_heads = nn.ModuleList([make_view_block() for _ in range(num_heads)])
        #         self.views_shared = None
        #     else:
        #         self.views_shared = make_view_block(self.view_hidden)
        #         self.views_heads = None      
        # else:
        #     self.views_shared = None
        #     self.views_heads = None


        
        self.view1_hidden= W  if self.use_viewdirs else 0

        def make_view1_block():
            # single layer: concat(feature, dir_embed) -> W
            return nn.ModuleList([nn.Linear(W + self.input_ch_views, self.view1_hidden)])
        
        if self.use_viewdirs:
            if branch_from in ('views1'):
                self.views1_heads = nn.ModuleList([make_view1_block() for _ in range(num_heads)])
                self.views1_shared = None
            else:
                self.views1_shared =nn.ModuleList(make_view1_block())
                self.views1_heads = None
                
            if branch_from in ('views1','views','feature'):
                self.views_heads = nn.ModuleList([nn.ModuleList([nn.Linear(W ,W//2)]) for _ in range(num_heads)])
                self.views_shared = None
            else:
                self.views_shared = nn.ModuleList(nn.ModuleList([nn.Linear(W ,W//2)]))
                self.views_heads = None
                
        else:
            self.views1_shared = None
            self.views1_heads = None
            self.views_shared = None
            self.views_heads = None

        self.view_hidden=W//2  if self.use_viewdirs else 0



        # RGB heads are always per-head in multi-domain setups
        if self.use_viewdirs:
            self.rgb_heads = nn.ModuleList([nn.Linear(self.view_hidden, 3) for _ in range(num_heads)])
        else:
            # If not using viewdirs, RGB comes straight from W features
            self.rgb_heads = nn.ModuleList([nn.Linear(W, 3) for _ in range(num_heads)])

    # ----- helpers -----
    @staticmethod
    def _apply_block(block, x):
        # block is ModuleList of linear layers with ReLU between
        for l in block:
            x = F.relu(l(x), inplace=True)
        return x

    def _route_linear_per_head(self, linears, x, head_idx):
        """Apply per-head Linear (out_features known from module)."""
        # head_idx: [N] long
        out_dim = linears[0].out_features
        out = x.new_zeros(x.shape[0], out_dim)
        for h in range(len(linears)):
            mask = (head_idx == h)
            if mask.any():
                out[mask] = linears[h](x[mask])
        return out

    def _route_viewblock_per_head(self, blocks, x, head_idx):
        """Apply per-head view block (ModuleList of layers)."""
        # get output dim by running 1 example through head 0
        with torch.no_grad():
            dummy = self._apply_block(blocks[0], x[:1])
            out_dim = dummy.shape[-1]
        out = x.new_zeros(x.shape[0], out_dim)
        for h in range(len(blocks)):
            mask = (head_idx == h)
            if mask.any():
                out[mask] = self._apply_block(blocks[h], x[mask])
        return out

    # ----- forward -----
    def forward(self, x, head_idx=None):
        """
        x: [N, input_ch (+ input_ch_views if use_viewdirs)]
        head_idx: None or [N] long (domain id per sample)
        """
        if head_idx is None:
            # default to head 0
            head_idx = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        # Split xyz / dir embeddings
        if self.use_viewdirs:
            x_pts, x_dirs = x[..., :self.input_ch], x[..., self.input_ch:]
        else:
            x_pts, x_dirs = x, None

        # Shared xyz trunk
        h = x_pts
        for i, l in enumerate(self.pts_linears):
            if i in self.skips:
                h = torch.cat([x_pts, h], -1)
            h = F.relu(l(h), inplace=True)
            

        # Alpha (σ)
        if self.alpha_shared is not None:
            alpha = self.alpha_shared(h)
        else:
            alpha = self._route_linear_per_head(self.alpha_heads, h, head_idx)

        # feature_linear
        if self.feature_heads is not None:
            feat = self._route_linear_per_head(self.feature_heads, h, head_idx)
        else:
            feat = self.feature_shared(h)

        # Color branch
        if self.use_viewdirs:
            v_in = torch.cat([feat, x_dirs], -1)
            if self.views_shared is not None:
                v = self._apply_block(self.views_shared, v_in)
            else:
                v = self._route_viewblock_per_head(self.views_heads, v_in, head_idx)
            # RGB per head
            rgb = self._route_linear_per_head(self.rgb_heads, v, head_idx)
        else:
            # No viewdirs: rgb directly from features
            rgb = self._route_linear_per_head(self.rgb_heads, feat, head_idx)

        raw = torch.cat([rgb, alpha], -1)  # [N, 4]
        if self.output_ch == 5:
            pad = torch.zeros_like(alpha)
            raw = torch.cat([raw, pad], -1)  # [N, 5]
        return raw


# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples