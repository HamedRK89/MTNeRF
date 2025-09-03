# grad_similarity_runner.py
import os, csv, time, math
import numpy as np
from tqdm import tqdm, trange

import torch
import torch.nn.functional as F

from splitless_nerf_model import SingleNeRF, render_rays, run_network, ndc_rays
from run_nerf_helpers import get_embedder, get_rays_np, img2mse, mse2psnr
from load_llff import load_llff_data

# --------------- device pick (cpu/cuda/mps) ---------------
def pick_device(prefer_mps=False):
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ---------------- CSV logger ----------------
class CSVLogger:
    def __init__(self, path, fieldnames):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.fieldnames = fieldnames
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()

    def write(self, row):
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            w.writerow(row)

# --------------- gradient utils ---------------
def flatten_or_zeros(t, like):
    if t is None:
        return torch.zeros_like(like).view(-1)
    return t.view(-1)

def grad_by_groups(loss, model, groups):
    """
    Compute grads w.r.t. a flat param list, grouped by `groups` dict(name -> [params]).
    Returns: dict(name -> flat_grad_tensor)
    """
    all_params = []
    index_slices = {}
    start = 0
    for gname, plist in groups.items():
        for p in plist:
            n = p.numel()
            index_slices.setdefault(gname, []).append(slice(start, start+n))
            all_params.append(p)
            start += n
    flat_like = torch.cat([p.detach().view(-1) for p in all_params])

    grads = torch.autograd.grad(
        loss, all_params, retain_graph=True, create_graph=False, allow_unused=True
    )
    flat_grads = torch.cat([flatten_or_zeros(g, p.detach()).view(-1) for g, p in zip(grads, all_params)])

    out = {}
    for gname, slices in index_slices.items():
        segs = [flat_grads[s] for s in slices]
        out[gname] = torch.cat(segs)
    return out

def cos_sim(a, b, eps=1e-12):
    return torch.dot(a, b) / (a.norm() * b.norm() + eps)

# --------------- data helpers ---------------
def rays_to_tensor(rays, images, sel_indices=None):
    """
    rays: [N, 2, H, W, 3] from get_rays_np (ro, rd)
    images: [N, H, W, 3]
    returns flattened [M, 3, 3] as in your code (ro, rd, rgb)
    """
    if sel_indices is not None:
        rays = rays[sel_indices]
        images = images[sel_indices]
    rays_rgb = np.concatenate([rays, images[:, None]], 1)     # [N, ro+rd+rgb, H, W, 3]
    rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4])            # [N, H, W, 3, 3]
    rays_rgb = np.reshape(rays_rgb, [-1,3,3])                 # [N*H*W, 3, 3]
    rays_rgb = rays_rgb.astype(np.float32)
    np.random.shuffle(rays_rgb)
    return rays_rgb

# --------------- main ---------------
def main():
    import argparse
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--datadir", type=str, required=True)
    p.add_argument("--factor", type=int, default=8)
    # model/render
    p.add_argument("--netdepth", type=int, default=8)
    p.add_argument("--netwidth", type=int, default=256)
    p.add_argument("--use_viewdirs", action="store_true")
    p.add_argument("--i_embed", type=int, default=0)
    p.add_argument("--multires", type=int, default=10)
    p.add_argument("--multires_views", type=int, default=4)
    p.add_argument("--N_samples", type=int, default=64)
    p.add_argument("--N_importance", type=int, default=0)
    p.add_argument("--perturb", type=float, default=1.0)
    p.add_argument("--raw_noise_std", type=float, default=0.0)
    p.add_argument("--white_bkgd", action="store_true")
    # training
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--N_rand", type=int, default=1024)
    p.add_argument("--lrate", type=float, default=5e-4)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--sim_batch", type=int, default=2048, help="rays per domain for gradient similarity")
    # logging
    p.add_argument("--basedir", type=str, default="./logs")
    p.add_argument("--expname", type=str, default="grad_sims")
    # device
    p.add_argument("--mps", action="store_true")
    args = p.parse_args()

    device = pick_device(prefer_mps=args.mps)
    print("Device:", device)

    # --------- Load LLFF with original + virtual (augmented) ---------
    images_o, poses_o, bds_o, images_v, poses_v, bds_v, render_poses, i_test = load_llff_data(
        args.datadir, args.factor, recenter=True, bd_factor=.75, spherify=False
    )
    num_o = images_o.shape[0]
    num_v = images_v.shape[0]
    H, W, focal = poses_o[0,:3,-1]
    H, W = int(H), int(W)
    K = np.array([[focal,0,0.5*W],[0,focal,0.5*H],[0,0,1]], dtype=np.float32)

    # splits (simple): hold out every 8th original image for test, validate is ignored here
    stride = 8
    i_test = np.arange(num_o)[::stride]
    i_train = np.array([i for i in range(num_o) if i not in i_test])

    # precompute rays
    rays_o_np = np.stack([get_rays_np(H, W, K, p[:3,:4]) for p in poses_o], 0) # [N,2,H,W,3]
    rays_v_np = np.stack([get_rays_np(H, W, K, p[:3,:4]) for p in poses_v], 0)

    # flatten pools
    rays_rgb_o = rays_to_tensor(rays_o_np, images_o, sel_indices=i_train)  # [*,3,3]
    rays_rgb_v = rays_to_tensor(rays_v_np, images_v)                       # [*,3,3]

    # tensors to device
    rays_rgb_o = torch.from_numpy(rays_rgb_o).to(device)
    rays_rgb_v = torch.from_numpy(rays_rgb_v).to(device)

    # --------- Embedders ---------
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    else:
        embeddirs_fn, input_ch_views = (None, 0)

    # --------- Model + optimizer ---------
    model = SingleNeRF(D=args.netdepth, W=args.netwidth,
                       input_ch=input_ch, input_ch_views=input_ch_views,
                       output_ch=(5 if args.N_importance>0 else 4),
                       skips=(4,), use_viewdirs=args.use_viewdirs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, betas=(0.9,0.999))

    # group params for per-block gradient similarity
    groups = {
        "trunk":     [p for n,p in model.named_parameters() if n.startswith("pts_linears")],
        "feature":   [p for n,p in model.named_parameters() if n.startswith("feature_linear")],
        "views":     [p for n,p in model.named_parameters() if n.startswith("views_linears")] if args.use_viewdirs else [],
        "rgb":       [p for n,p in model.named_parameters() if n.startswith("rgb_linear")],
        "alpha":     [p for n,p in model.named_parameters() if n.startswith("alpha_linear")],
    }

    # renderer closures
    def network_query_fn(pts, viewdirs, network_fn):
        return run_network(pts, viewdirs, network_fn, embed_fn, embeddirs_fn, netchunk=1024*64)

    def pack_batch(batch):  # batch: [B,3,3] -> (ray_batch tensor)
        rays_o = batch[:,0,:]; rays_d = batch[:,1,:]; target = batch[:,2,:]
        near = torch.zeros_like(rays_o[..., :1]); far = torch.ones_like(rays_o[..., :1])
        ray_batch = torch.cat([rays_o, rays_d, near, far], dim=-1)
        return ray_batch, target

    # CSV logger
    out_csv = os.path.join(args.basedir, args.expname, "grad_similarity.csv")
    logger = CSVLogger(out_csv, ["step","block","cosine","g_orig_norm","g_aug_norm"])

    # training loops
    ptr_o = 0; ptr_v = 0
    B = args.N_rand
    B2 = args.sim_batch

    for step in trange(1, args.iters+1):
        # ---- sample a mixed training batch (half/half) ----
        if ptr_o + B//2 >= rays_rgb_o.shape[0]:
            perm = torch.randperm(rays_rgb_o.shape[0], device=device)
            rays_rgb_o = rays_rgb_o[perm]; ptr_o = 0
        if ptr_v + B - B//2 >= rays_rgb_v.shape[0]:
            perm = torch.randperm(rays_rgb_v.shape[0], device=device)
            rays_rgb_v = rays_rgb_v[perm]; ptr_v = 0

        batch = torch.cat([rays_rgb_o[ptr_o:ptr_o+B//2],
                           rays_rgb_v[ptr_v:ptr_v+(B-B//2)]], dim=0)
        ptr_o += B//2; ptr_v += (B-B//2)

        ray_batch, target = pack_batch(batch)
        # no ndc for LLFF forward-facing scenes? set if you normally use it
        # (we keep ndc off here because near/far are [0,1] already)
        ret = render_rays(ray_batch, model, network_query_fn,
                          N_samples=args.N_samples, retraw=False,
                          lindisp=False, perturb=args.perturb,
                          N_importance=args.N_importance, network_fine=None,
                          white_bkgd=args.white_bkgd, raw_noise_std=args.raw_noise_std,
                          use_viewdirs=args.use_viewdirs)
        rgb = ret["rgb_map"]
        loss = img2mse(rgb, target)
        psnr = mse2psnr(loss)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            tqdm.write(f"[train] step {step}  loss {loss.item():.5f}  psnr {psnr.item():.2f}")

        # ---- gradient similarity eval ----
        if step % args.eval_every == 0:
            model.zero_grad(set_to_none=True)

            # sample pure original & pure augmented eval batches
            if ptr_o + B2 >= rays_rgb_o.shape[0]:
                perm = torch.randperm(rays_rgb_o.shape[0], device=device)
                rays_rgb_o = rays_rgb_o[perm]; ptr_o = 0
            if ptr_v + B2 >= rays_rgb_v.shape[0]:
                perm = torch.randperm(rays_rgb_v.shape[0], device=device)
                rays_rgb_v = rays_rgb_v[perm]; ptr_v = 0

            batch_o = rays_rgb_o[ptr_o:ptr_o+B2]; ptr_o += B2
            batch_v = rays_rgb_v[ptr_v:ptr_v+B2]; ptr_v += B2

            rb_o, tgt_o = pack_batch(batch_o)
            rb_v, tgt_v = pack_batch(batch_v)

            # turn off jitter/noise during measurement to reduce variance
            common_kwargs = dict(N_samples=args.N_samples, retraw=False,
                                 lindisp=False, perturb=0.0,
                                 N_importance=args.N_importance, network_fine=None,
                                 white_bkgd=args.white_bkgd, raw_noise_std=0.0,
                                 use_viewdirs=args.use_viewdirs)

            out_o = render_rays(rb_o, model, network_query_fn, **common_kwargs)
            out_v = render_rays(rb_v, model, network_query_fn, **common_kwargs)
            loss_o = img2mse(out_o["rgb_map"], tgt_o)
            loss_v = img2mse(out_v["rgb_map"], tgt_v)

            # grads at SAME weights for each domain
            g_o = grad_by_groups(loss_o, model, groups)
            g_v = grad_by_groups(loss_v, model, groups)

            for block in ["trunk","feature","views","rgb","alpha"]:
                if len(groups[block]) == 0:
                    continue
                co = cos_sim(g_o[block], g_v[block]).item()
                logger.write({
                    "step": step,
                    "block": block,
                    "cosine": f"{co:.6f}",
                    "g_orig_norm": f"{g_o[block].norm().item():.6f}",
                    "g_aug_norm": f"{g_v[block].norm().item():.6f}",
                })
            tqdm.write(f"[sim] step {step}: " +
                       ", ".join([f"{b} {cos_sim(g_o[b], g_v[b]).item():.2f}"
                                  for b in ["trunk","feature","views","rgb","alpha"]
                                  if len(groups[b])>0]))

if __name__ == "__main__":
    main()
