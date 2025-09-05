import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, head_idx=None, netchunk=1024*64):
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)       # [N_rays,N_samples,3]
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    # Prepare head indices per sample
    head_idx_flat = None
    if head_idx is not None:
        # head_idx: [N_rays] -> [N_rays,N_samples] -> [N_rays*N_samples]
        n_rays, n_samples = inputs.shape[0], inputs.shape[1]
        head_idx_flat = head_idx[:, None].expand(n_rays, n_samples).reshape(-1).to(torch.long)

    # Chunked forward with head selection
    outs = []
    N = embedded.shape[0]
    for i in range(0, N, netchunk):
        hi = head_idx_flat[i:i+netchunk] if head_idx_flat is not None else None
        outs.append(fn(embedded[i:i+netchunk], head_idx=hi))
    outputs_flat = torch.cat(outs, dim=0)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    domains = None
    if c2w is not None:
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        if isinstance(rays, (list, tuple)):
            if len(rays) == 3:
                rays_o, rays_d, domains = rays
            else:
                rays_o, rays_d = rays
        else:
            rays_o, rays_d = rays

    viewdirs = None
    if use_viewdirs:
        vdirs = rays_d
        if c2w_staticcam is not None:
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        vdirs = vdirs / torch.norm(vdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(vdirs, [-1, 3]).float()

    sh = rays_d.shape
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near_t = near * torch.ones_like(rays_d[..., :1])
    far_t  = far  * torch.ones_like(rays_d[..., :1])
    ray_batch = torch.cat([rays_o, rays_d, near_t, far_t], dim=-1)  # [N, 8]

    if use_viewdirs:
        ray_batch = torch.cat([ray_batch, viewdirs], dim=-1)        # [N, 11]

    if domains is None:
        dom = torch.zeros(rays_o.shape[0], dtype=torch.long, device=rays_o.device)
    else:
        dom = torch.reshape(domains, [-1]).long().to(rays_o.device)
    ray_batch = torch.cat([ray_batch, dom.float().unsqueeze(-1)], dim=-1)  # [N, 9 or 12]

    all_ret = batchify_rays(ray_batch, chunk, use_viewdirs=use_viewdirs, **kwargs)

    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'depth_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                use_viewdirs=False):

    N_rays = ray_batch.shape[0]
    rays_o = ray_batch[:, 0:3]
    rays_d = ray_batch[:, 3:6]
    bounds = torch.reshape(ray_batch[:, 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]

    base = 8
    if use_viewdirs:
        viewdirs = ray_batch[:, base:base+3]
        idx_domain = base + 3
    else:
        viewdirs = None
        idx_domain = base

    domain_ids = ray_batch[:, idx_domain].long()

    t_vals = torch.linspace(0., 1., steps=N_samples, device=rays_o.device)
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * t_vals
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        mids  = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape, device=z_vals.device)
        if pytest:
            np.random.seed(0)
            t_rand = torch.tensor(np.random.rand(*list(z_vals.shape)),
                                  device=z_vals.device, dtype=z_vals.dtype)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

    # Coarse
    raw = network_query_fn(pts, viewdirs, domain_ids, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest
    )

    # Fine
    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples  = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
        z_samples  = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, domain_ids, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest
        )

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map, 'depth_map': depth_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
    return ret


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, depth, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)

    output_ch = 5 if args.N_importance > 0 else 4
    model = SplitLayerNeRF(
    D=args.netdepth, W=args.netwidth,
    input_ch=input_ch, input_ch_views=input_ch_views,
    output_ch=output_ch, skips=[4], use_viewdirs=args.use_viewdirs,
    num_heads=2,
    branch_from=args.branch_from,   # <-- new
    share_alpha=args.share_alpha    # <-- optional
).to(device)

    model_fine = None
    if args.N_importance > 0:
        model_fine = SplitLayerNeRF(
            D=args.netdepth_fine, W=args.netwidth_fine,
            input_ch=input_ch, input_ch_views=input_ch_views,
            output_ch=output_ch, skips=[4], use_viewdirs=args.use_viewdirs,
            num_heads=2,
            branch_from=args.branch_from,
            share_alpha=args.share_alpha
        ).to(device)




    # IMPORTANT: signature is (inputs, viewdirs, head_idx, network_fn)
    network_query_fn = lambda inputs, viewdirs, head_idx, network_fn: run_network(
        inputs, viewdirs, network_fn,
        embed_fn=embed_fn, embeddirs_fn=embeddirs_fn,
        head_idx=head_idx, netchunk=args.netchunk
    )

    # optimizer
    grad_vars = list(model.parameters()) + ([] if model_fine is None else list(model_fine.parameters()))
    optimizer = torch.optim.Adam(grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    # Load checkpoints (if any)
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        expdir = os.path.join(basedir, expname)
        os.makedirs(expdir, exist_ok=True)
        ckpts = [os.path.join(expdir, f) for f in sorted(os.listdir(expdir)) if f.endswith('.tar')]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)
        start = ckpt['global_step']
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None and 'network_fine_state_dict' in ckpt and ckpt['network_fine_state_dict'] is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    big = torch.tensor([1e10], device=z_vals.device, dtype=z_vals.dtype).expand(dists[..., :1].shape)
    dists = torch.cat([dists, big], -1)

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape, device=raw.device) * raw_noise_std
        if pytest:
            np.random.seed(0)
            n = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.tensor(n, device=raw.device, dtype=raw.dtype)

    alpha = raw2alpha(raw[..., 3] + noise, dists)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device),
                                               1. - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(torch.tensor(1e-10, device=depth_map.device, dtype=depth_map.dtype),
                              depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map






def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--landa", type=float, default=1.0, 
                        help='Regularization parameter for downweighting augmented images loss due to their groundtruth noise')
    #multitask branchout 
    parser.add_argument("--branch_from", type=str, default="heads",
                        choices=["heads","views","feature"],
                        help="split point after the shared 8-layer trunk")
    
    #If you want σ to be shared in any of these (alpha_heads=alpha_linear : W->1)
    parser.add_argument("--share_alpha", action="store_true",
                        help="keep σ head shared even when splitting earlier")


    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=200000, 
                        help='frequency of render_poses video saving')
    # select GPU
    parser.add_argument("--gpu_id", type=str, default=0, required=False,
                        help="gpu id to use")
    # test set options
    parser.add_argument('--i_test', nargs='+', type=int, default=None,
                        help='A list of integers')
    
    # depth supervision
    parser.add_argument("--depth_supervision", type=bool, default=False,
                        help='True if depth data is available and will be used during training')
    # debug
    parser.add_argument("--debug", action='store_true')

    # mac users
    parser.add_argument("--mps", action='store_true',
                        help="For Mac users if they want to use MPS GPU")

    return parser

def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images_orig, poses_orig, bds_orig, images_virtual, poses_virtual, bds_virtual, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)

        num_orig = images_orig.shape[0]
        num_virtual=images_virtual.shape[0]
        total_num = num_orig + num_virtual

        print(f'Number of original images:  {num_orig} \n Number of virtual images: {num_virtual}')
        print(f'Original images shape: {images_orig.shape}, Original poses shape: {poses_orig.shape}')
        print(f'Virtual images shape: {images_virtual.shape}, Virtual poses shape: {poses_virtual.shape}')

        if args.depth_supervision:
            depths = _load_depth_data(args.datadir, args.factor, load_depth=True) 
            print(f'depth shape: {depths.shape}') # numpy array

        hwf = poses_orig[0,:3,-1]
        poses_orig = poses_orig[:,:3,:4]
        poses_virtual = poses_virtual[:,:3,:4]
        print("Original poses shape:----------> ", poses_orig.shape)
        print("Virtual poses shape:----------> ", poses_virtual.shape)
        #print(f'Loaded llff, images shape: {images.shape}, render poses shape: {render_poses.shape}, hwf: {hwf}, data dir: {args.datadir}')
        
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images_orig.shape[0])[::args.llffhold]
        
        if args.i_test is not None:
            i_test = args.i_test
        
        i_val = i_test
        print("testxxxxxxxxxxxxxxxxxxxxxxxxx", i_test, i_val)
        i_train = np.array([i for i in np.arange(num_orig) if
                        (i not in i_test and i not in i_val)])
        print("i_train:", i_train)
        num_test = len(i_test)
        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = min(bds_orig.min(), bds_virtual.min()) * .9
            far = max(bds_orig.max(), bds_virtual.max()) * 1.0      
      
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses_orig[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars_orig, optimizer= create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand_orig = int(args.N_rand * ((num_orig-num_test)/(total_num-num_test)))
    N_rand_virtual = args.N_rand - N_rand_orig
    #print('num_orig: ', num_orig,"num_virtual",num_virtual, "total_num: ", total_num, "N_rand: ",args.N_rand,"num_rand_orig:", N_rand_orig)
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        # print("****************Poses orig************ :\n ", poses_orig)
        # print("****************Poses virtual************ :\n ", poses_virtual)
        rays_orig = np.stack([get_rays_np(H, W, K, p) for p in poses_orig[:,:3,:4]], 0) # [N_orig, ro+rd, H, W, 3]
        rays_virtual = np.stack([get_rays_np(H, W, K, p) for p in poses_virtual[:,:3,:4]], 0) # [N_virtual, ro+rd, H, W, 3]

        # Rays shape:   [N, ro+rd, H, W, 3] --> Rays_orig shape: ?? // Rays_virtual shape: ??
        # Images shape: [N, H, W, 3] --> Images_orig shape: ?? // Images_virtual shape: ??
        print(f"SHAPES:\n\tRays_orig Shape: {rays_orig.shape}, Images_orig Shape: {images_orig.shape}")
        print(f"SHAPES:\n\tRays_virtual Shape: {rays_virtual.shape}, Images_virtual Shape: {images_virtual.shape}")
        '''
        if args.debug:
            print(f"SHAPES:\n\tRays Shape: {rays.shape}, Images Shape: {images.shape}")
            if args.depth_supervision:
                print(f"Depths Shape: {depths.shape}")
        '''
        if not args.depth_supervision:
            print('done, concats')
            rays_rgb_orig = np.concatenate([rays_orig, images_orig[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
            rays_rgb_orig = np.transpose(rays_rgb_orig, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
            rays_rgb_orig = np.stack([rays_rgb_orig[i] for i in i_train], 0) # train images only, Needs to be corrected and generalized later!
            rays_rgb_orig = np.reshape(rays_rgb_orig, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
            rays_rgb_orig = rays_rgb_orig.astype(np.float32)
            np.random.shuffle(rays_rgb_orig)

            rays_rgb_virtual = np.concatenate([rays_virtual, images_virtual[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
            rays_rgb_virtual = np.transpose(rays_rgb_virtual, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
            rays_rgb_virtual = np.stack([rays_rgb_virtual[i] for i in range(num_virtual)], 0) # train images only
            rays_rgb_virtual = np.reshape(rays_rgb_virtual, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
            rays_rgb_virtual = rays_rgb_virtual.astype(np.float32)
            np.random.shuffle(rays_rgb_virtual)
        
        ''''
        else:
            # First concatenate rays, images, and depths
            # Add dimensions to images and depths to match rays' structure
            images_expanded = images[:, None, ...]  # [N, 1, H, W, 3]
            depths_expanded = depths[:, None, ..., None]  # [N, 1, H, W, 1]
            
            # Repeat depth values 3 times to match the last dimension size
            depths_repeated = np.repeat(depths_expanded, 3, axis=-1)  # [N, 1, H, W, 3]
            # Now concatenate all components
            rays_rgbd = np.concatenate([rays, images_expanded, depths_repeated], axis=1)  # [N, 4, H, W, 3]
            
            # Continue with the original reshaping pipeline
            rays_rgbd = np.transpose(rays_rgbd, [0, 2, 3, 1, 4])  # [N, H, W, 4, 3]
            rays_rgbd = np.stack([rays_rgbd[i] for i in i_train], 0)  # train images only
            rays_rgbd = np.reshape(rays_rgbd, [-1, 4, 3])  # [(N-1)*H*W, 4, 3]
            rays_rgbd = rays_rgbd.astype(np.float32) # Before dtype was: float64
            np.random.shuffle(rays_rgbd)
            '''

        print('shuffle rays')
        print('done')
        i_orig=0
        i_virtual=0

    # Move training data to GPU
    if use_batching:
        images_orig = torch.Tensor(images_orig).to(device)
        images_virtual = torch.Tensor(images_virtual).to(device)
        if args.depth_supervision:
            depths = torch.Tensor(depths).to(device)
    poses_orig = torch.Tensor(poses_orig).to(device)
    poses_virtual = torch.Tensor(poses_virtual).to(device)
    if use_batching:
        if not args.depth_supervision:
            rays_rgb_orig = torch.Tensor(rays_rgb_orig).to(device)
            rays_rgb_virtual = torch.Tensor(rays_rgb_virtual).to(device)
            
        else:
            rays_rgbd = torch.Tensor(rays_rgbd).to(device)
            

    N_iters = 200000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    #writer = SummaryWriter(os.path.join(basedir, expname, 'tensorboard'))
    # print("KWARG_TRAIN", render_kwargs_train)
    # print("KWARG_TEST", render_kwargs_test)
    print("rays_rgb_orig:",rays_rgb_orig.shape)
    print("rays_rgb_virtual:",rays_rgb_virtual.shape)

    
    #rays_rgb=torch.cat([rays_rgb_orig,rays_rgb_virtual],dim=0)
    N_rand=args.N_rand
    start = start + 1
    i_batch=0
    flag_virtual=1
    flag_orig=1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            if not args.depth_supervision:

                


            # ---- PROPORTIONAL COMBINATION (single render) ----
                # how many from orig this step (fixed proportion)
                n_o_desired = N_rand_orig
                rem_o = rays_rgb_orig.shape[0] - i_orig
                rem_v = rays_rgb_virtual.shape[0] - i_virtual

                # take what we can
                n_o = min(n_o_desired, rem_o)
                n_v = min(N_rand - n_o, rem_v)

                # top up if short so total ≈ N_rand
                short = N_rand - (n_o + n_v)
                if short > 0:
                    extra_o = min(short, rem_o - n_o); n_o += extra_o; short -= extra_o
                if short > 0:
                    extra_v = min(short, rem_v - n_v); n_v += extra_v; short -= extra_v
                # if short > 0 here, final batch is just smaller than N_rand

                parts, wparts=[], []
                if n_o > 0:
                    bo = rays_rgb_orig[i_orig:i_orig + n_o]; i_orig += n_o     # [n_o,3,3]
                    parts.append(bo)
                    wparts.append(torch.ones(n_o, device=bo.device))           # weight 1.0
                if n_v > 0:
                    bv = rays_rgb_virtual[i_virtual:i_virtual + n_v]; i_virtual += n_v
                    parts.append(bv)
                    wparts.append(torch.full((n_v,), args.landa, device=bv.device))  # weight λ

                batch = torch.cat(parts, dim=0)          # [B,3,3]
                wts   = torch.cat(wparts, dim=0)         # [B]

                # to [2,B,3] rays + [B,3] target
                b = batch.transpose(0, 1)                # [3,B,3]
                batch_rays, target = b[:2], b[2]

                bo=batch.transpose(0,1)
                bv=batch.transpose(0,1)
                batch_rays_o, target_o = bo[:2], bo[2]
                batch_rays_v, target_v = bv[:2], bv[2]

                # domains: 0 for original, 1 for virtual, matching the concatenation order
                dom_parts = []
                if n_o > 0:
                    dom_parts.append(torch.zeros(n_o, dtype=torch.long, device=batch.device))
                if n_v > 0:
                    dom_parts.append(torch.ones(n_v, dtype=torch.long, device=batch.device))
                domains = torch.cat(dom_parts, dim=0)     # [B]

                # end-of-epoch reshuffle (both pools exhausted)
                if i_orig >= rays_rgb_orig.shape[0] and i_virtual >= rays_rgb_virtual.shape[0]:
                    print("Shuffle data after an epoch!")
                    perm_o = torch.randperm(rays_rgb_orig.shape[0], device=rays_rgb_orig.device)
                    perm_v = torch.randperm(rays_rgb_virtual.shape[0], device=rays_rgb_virtual.device)
                    rays_rgb_orig    = rays_rgb_orig[perm_o]
                    rays_rgb_virtual = rays_rgb_virtual[perm_v]
                    i_orig = 0
                    i_virtual = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

#         rgb, disp, acc, depth, extras = render(
#     H, W, K, chunk=args.chunk,
#     rays=(batch_rays[0], batch_rays[1], domains),
#     verbose=i < 10, retraw=True, **render_kwargs_train
# )
        rgb_o, disp, acc, depth, extras_o = render(
    H, W, K, chunk=args.chunk,
    rays=(batch_rays_o[0], batch_rays_o[1], domains),
    verbose=i < 10, retraw=True, **render_kwargs_train
)
        rgb_v, disp, acc, depth, extras_v = render(
    H, W, K, chunk=args.chunk,
    rays=(batch_rays_v[0], batch_rays_v[1], domains),
    verbose=i < 10, retraw=True, **render_kwargs_train
)



        optimizer.zero_grad()
        if not args.depth_supervision:

            # per_ray = ((rgb - target) ** 2).mean(dim=1)          # [B]
            # loss = (wts * per_ray).sum() / wts.sum()
            # psnr= mse2psnr(loss)


            # if 'rgb0' in extras:
            #     per_ray0 = ((extras['rgb0'] - target) ** 2).mean(dim=1)
            #     loss += (wts * per_ray0).sum() / wts.sum()




            loss_o=((rgb_o - target_o) ** 2).mean(dim=1) 
            loss_v=((rgb_v - target_v) ** 2).mean(dim=1) 
            loss=loss_o+args.landa*loss_v
            psnr=mse2psnr(loss)

            if 'rgb0' in extras_o:
                loss_rgb0_o=((extras_o['rgb0'] - target_o) ** 2).mean(dim=1)
                loss+=loss_rgb0_o

            if 'rgb0' in extras_v:
                loss_rgb0_v=((extras_v['rgb0'] - target_v) ** 2).mean(dim=1)
                loss+=args.landa*loss_rgb0_v
                

            # PSNR for logging (fine-only, unweighted to avoid mix-induced jumps)
            # with torch.no_grad():
            #     psnr = mse2psnr(((rgb - target) ** 2).mean())

        else:
            # print(f"----------------------------\n RGB: {rgb} \n target_RGB: {target_rgb} \n  epth: {depth} \n target_d:{target_d}")
            #rendered_depth = 1. / torch.clamp(disp, min=1e-6)
            depth_loss = 0
            # 1. Photometric (RGB) loss (original)
            img_loss = img2mse(rgb, target_rgb)  # Compare with target_rgb (not target_s)
            loss = img_loss

            # 2. Depth loss (new)
            # Only apply where ground truth depth is valid (depth > 0)
            # valid_depth_mask = (target_d > 0).float()  # [B, 1]
            # depth_loss = F.mse_loss(disp * valid_depth_mask, target_d * valid_depth_mask)
            if not isinstance(depth, torch.Tensor):
                depth = torch.tensor(depth, device=target_d.device, dtype=target_d.dtype)
            
            if depth.dim() > 1:
                # If using per-sample depths (incorrect), use volumetric rendered depth
                depth = extras['depth_map']  # Get from render outputs
                depth = depth.squeeze(-1) if depth.shape[-1] == 1 else depth  # [N_rays]
                target_d = target_d.squeeze(-1) if target_d.shape[-1] == 1 else target_d  # [N_rays]

                # Verify device matching
                depth = depth.to(device=target_d.device)
            target_d = target_d.squeeze(-1)  # From [N_rays, 1] to [N_rays]

            # Verify shapes match
            # print(f"*_*_*_*_*_*_*_*_*Depth shape: {depth.shape}, Target shape: {target_d.shape}")
            # Ensure both tensors on same device
            depth = depth.to(target_d.device)
            disp = disp.to(target_d.device)

            # Verify devices match
            # print(f"Depth device: {depth.device}, Target device: {target_d.device}")
            # Should output: cuda:0 for both (or cpu for both)
            # print(f"----------------------------\n RGB: {rgb} \n target_RGB: {target_rgb} \n Disp: {disp} \n target_d:{target_d}")              
            # Todo: Normalize disp!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #print(f"---------------********************------------------ Disp_shape: {disp.shape}, \n target_d: {target_d.shape}") #torch.Size([1024])
            disp_norm = (disp - disp.min()) / (disp.max() - disp.min() + 1e-8)
            #print(f"---------------********************------------------ \n Disp: {disp_norm.min(), disp_norm.max()}, \n target_d: {target_d.min(), target_d.max()}")
            depth_loss = img2mse(disp, target_d)
            # print(f"-----disp----\n:", max(disp))
            # print(f"-----target_d----\n:", max(target_d))
            # depth_loss = img2mse(1. / (torch.clamp(depth, min=1e-6)), (target_d))
            # depth_loss = torch.mean((depth - target_d) ** 2)
            
            depth_weight = 0.1  # Start with lower weight

            # if i > 100000:  # Optionally increase weight later
            #     depth_weight = 0.11


            loss = loss + depth_weight * depth_loss # weight hyperparameter

            # 3. Optional: Disparity regularization (original)
            trans = extras['raw'][...,-1]  # transparency

            # 4. Coarse network loss (if used)
            '''
            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_rgb)
                loss = loss + img_loss0
                # Optional: Add depth loss for coarse network
                if 'depth0' in extras:
                    depth_loss0 = F.mse_loss(extras['depth0'] * valid_depth_mask,
                                        target_d * valid_depth_mask)
                    loss = loss + args.depth_weight * depth_loss0
            
            # Metrics
            psnr = mse2psnr(img_loss)
            if 'depth0' in extras:
                depth_rmse = torch.sqrt(depth_loss0).item()  # For logging
            '''
        '''
        # Log losses and metrics
        writer.add_scalar('Loss/total', loss.item(), i)
        writer.add_scalar('Loss/img', img_loss.item(), i)
        writer.add_scalar('Metrics/PSNR', psnr.item(), i)
        
        if args.depth_supervision:
            writer.add_scalar('Loss/depth', depth_loss.item(), i)
            #writer.add_scalar('Metrics/depth_rmse', depth_rmse, i)  # if available
            if i % args.i_img == 0:
                writer.add_image('Depth/Predicted', depth_map, i)
                writer.add_image('Depth/Ground_Truth', target_d, i)
        
        if 'rgb0' in extras:
            writer.add_scalar('Loss/img0', img_loss0.item(), i)
            writer.add_scalar('Metrics/PSNR0', psnr0.item(), i)
        '''
        # Log learning rate
        #writer.add_scalar('Learning_rate', new_lrate, i)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses_orig[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses_orig[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images_orig[i_test], savedir=testsavedir)
            print('Saved test set')


    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        global_step += 1


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_dtype(torch.float32)
    train()
