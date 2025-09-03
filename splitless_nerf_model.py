# splitless_nerf_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Plain NeRF (no branches)
# ------------------------------
class SingleNeRF(nn.Module):
    """
    8-layer xyz trunk with skip at layer 4, + feature_linear, + (views->rgb) path, + alpha head.
    Forward expects concatenated embedded [x_embed, (dir_embed if use_viewdirs)].
    """
    def __init__(self, D=8, W=256,
                 input_ch=63,
                 input_ch_views=27,
                 output_ch=4,
                 skips=(4,),
                 use_viewdirs=True):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = set(skips or [])
        self.use_viewdirs = use_viewdirs
        self.output_ch = output_ch

        # xyz trunk
        self.pts_linears = nn.ModuleList()
        self.pts_linears.append(nn.Linear(input_ch, W))
        for i in range(1, D):
            in_ch = W + (input_ch if i in self.skips else 0)
            self.pts_linears.append(nn.Linear(in_ch, W))

        # sigma (alpha)
        self.alpha_linear = nn.Linear(W, 1)

        # feature bottleneck (shared)
        self.feature_linear = nn.Linear(W, W)

        # view branch (shared)
        if self.use_viewdirs:
            self.view_hidden = W // 2
            # concat(feature, dir_embed) -> hidden
            self.views_linears = nn.ModuleList([nn.Linear(W + input_ch_views, self.view_hidden)])
            self.rgb_linear = nn.Linear(self.view_hidden, 3)
        else:
            self.views_linears = None
            self.rgb_linear = nn.Linear(W, 3)

    def _apply_block(self, block, x):
        for l in block:
            x = F.relu(l(x), inplace=True)
        return x

    def forward(self, x, head_idx=None):
        # x : [N, input_ch + (input_ch_views if use_viewdirs)]
        if self.use_viewdirs:
            x_pts = x[..., :self.input_ch]
            x_dirs = x[..., self.input_ch:]
        else:
            x_pts = x
            x_dirs = None

        h = x_pts
        for i, l in enumerate(self.pts_linears):
            h = F.relu(l(h), inplace=True)
            if i in self.skips:
                h = torch.cat([x_pts, h], dim=-1)

        sigma = self.alpha_linear(h)                    # [N,1]
        feat  = self.feature_linear(h)                  # [N,W]

        if self.use_viewdirs:
            v_in = torch.cat([feat, x_dirs], dim=-1)    # [N, W+input_ch_views]
            v    = self._apply_block(self.views_linears, v_in)
            rgb  = self.rgb_linear(v)                   # [N,3]
        else:
            rgb  = self.rgb_linear(feat)                # [N,3]

        raw = torch.cat([rgb, sigma], dim=-1)           # [N,4]
        if self.output_ch == 5:
            raw = torch.cat([raw, torch.zeros_like(sigma)], dim=-1)
        return raw

# ------------------------------
# Rendering utilities
# ------------------------------
def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0., white_bkgd=False):
    # raw: [N_rays, N_samples, 4 or 5]; last dim [r,g,b,sigma,(pad)]
    device = raw.device
    rgb   = torch.sigmoid(raw[..., :3])

    sigma_a = raw[..., 3]
    if raw_noise_std > 0.:
        sigma_a = sigma_a + torch.randn_like(sigma_a) * raw_noise_std
    sigma_a = F.relu(sigma_a)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    alpha = 1. - torch.exp(-sigma_a * dists)
    T = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    weights = alpha * T

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals, dim=-1)
    acc_map = torch.sum(weights, dim=-1)
    disp_map = 1. / torch.clamp(depth_map / torch.sum(weights, dim=-1), min=1e-10)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map

def batchify(fn, chunk):
    if chunk is None:
        return fn
    def ret(inputs):
        outs = []
        for i in range(0, inputs.shape[0], chunk):
            outs.append(fn(inputs[i:i+chunk]))
        return torch.cat(outs, dim=0)
    return ret

def run_network(pts, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    pts_flat = torch.reshape(pts, [-1, pts.shape[-1]])
    embedded = embed_fn(pts_flat)

    if viewdirs is not None and viewdirs.shape[-1] > 0:
        # expand from [N_rays, 3] -> [N_rays, N_samples, 3]
        input_dirs = viewdirs[:, None, :].expand(pts.shape)  
        input_dirs = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(pts.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


@torch.no_grad()
def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # identical to the original NeRF NDC transform
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    o0, o1, o2 = rays_o[...,0]/rays_o[...,2], rays_o[...,1]/rays_o[...,2], torch.ones_like(rays_o[...,2])
    d0, d1, d2 = (rays_d[...,0]/rays_d[...,2] - o0), (rays_d[...,1]/rays_d[...,2] - o1), torch.zeros_like(rays_d[...,2])
    rays_o_ndc = torch.stack([o0, o1, o2], -1)
    rays_d_ndc = torch.stack([d0, d1, d2], -1)
    return rays_o_ndc, rays_d_ndc

def render_rays(ray_batch, network_fn, network_query_fn, N_samples,
                retraw=False, lindisp=False, perturb=0., N_importance=0,
                network_fine=None, white_bkgd=False, raw_noise_std=0.,
                use_viewdirs=False):

    N_rays = ray_batch.shape[0]
    rays_o = ray_batch[:, 0:3]
    rays_d = ray_batch[:, 3:6]
    bounds = ray_batch[:, 6:8].reshape(-1,1,2)
    near, far = bounds[...,0], bounds[...,1]

    base = 8
    viewdirs = ray_batch[:, base:base+3] if use_viewdirs else None

    t_vals = torch.linspace(0., 1., steps=N_samples, device=rays_o.device)
    z_vals = near * (1.-t_vals) + far * t_vals if not lindisp else 1. / (1./near * (1.-t_vals) + 1./far * t_vals)
    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        mids = 0.5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        z_vals = lower + (upper-lower)*torch.rand_like(z_vals)

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[..., :, None]
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd)

    if N_importance > 0:
        z_vals_mid = 0.5 * (z_vals[...,1:] + z_vals[...,:-1])
        from run_nerf_helpers import sample_pdf
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.))
        z_samples = z_samples.detach()
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[..., :, None]
        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map, 'depth_map': depth_map}
    if retraw:
        ret['raw'] = raw
    return ret
