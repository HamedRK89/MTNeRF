import numpy as np
import os, imageio
from shutil import copy2
from subprocess import check_output
from pathlib import Path
from PIL import Image
import imageio.v3 as iio

########## Slightly modified version of LLFF data loading code 
##########  see https://github.com/Fyusion/LLFF for original

def resize(imgs_dir, size, factor):
    for image in os.listdir(imgs_dir):
        if image.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_dir = os.path.join(imgs_dir, image)
            im = Image.open(image_dir)
            new_size = (int(size[1] / factor), int(size[0] / factor))  # Ensure the new size is a tuple of integers
            im = im.resize(new_size)
            im.save(image_dir) 

def _minify(basedir, factor, load_depth=False):
    # New _minify implementation
    print("minifiying iamges and depth maps (if load_depth=True)", factor, basedir)

    if not load_depth:
        needtoload = False
        imgdir = os.path.join(basedir, f'images_{factor}')
        
        if not os.path.exists(imgdir):
            needtoload = True
        if not needtoload:
            return
        
        imgdir = os.path.join(basedir, 'images')
        imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'jpeg', 'png', 'PNG']])]
        #imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'jpeg', 'png', 'PNG']])]
        imgdir_orig = imgdir

        name = f'images_{factor}'
        imgdir = os.path.join(basedir, name)

        os.makedirs(imgdir, exist_ok=True)

        for item in os.listdir(imgdir_orig):
            source = os.path.join(imgdir_orig, item)
            destination = os.path.join(imgdir, item)
            copy2(source, destination)
            #print(f"copying {item} to {destination}")

        

        ext = imgs[0].split('.')[-1]
        #ext_depth = depths[0].split('.')[-1]
        shape = imageio.imread(imgs[0]).shape
        #print("shape", shape)
        #print("shape 0", shape[0])
        #print("shape 1", shape[1])
        #print("factor", factor)
        #Resize the destination images!!!!
        resize(imgdir, shape, factor)
        

    if load_depth:
        depthdir = os.path.join(basedir, f'depths_{factor}')
        depthdir = os.path.join(basedir, 'depths')
        depths = [os.path.join(depthdir, f) for f in sorted(os.listdir(depthdir)) if any([f.endswith(ex) for ex in ['png', 'PNG']])]
        depthdir_orig = depthdir

        name_depth = f'depths_{factor}'
        depthdir = os.path.join(basedir, name_depth)

        os.makedirs(depthdir, exist_ok=True)

        for item in os.listdir(depthdir_orig):
            source = os.path.join(depthdir_orig, item)
            destination = os.path.join(depthdir, item)
            copy2(source, destination)
        shape = imageio.imread(depths[0]).shape
        resize(depthdir, shape, factor)

'''
def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
'''          
        
        
        
def  _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    
    poses_arr_orig = np.load(os.path.join(basedir, 'poses_bounds_orig.npy'))
    poses_orig = poses_arr_orig[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds_orig = poses_arr_orig[:, -2:].transpose([1,0])
    
    poses_arr_virtual = np.load(os.path.join(basedir, 'poses_bounds_virtual.npy'))
    poses_virtual = poses_arr_virtual[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds_virtual = poses_arr_virtual[:, -2:].transpose([1,0])

    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factor)
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if (poses_orig.shape[-1] + poses_virtual.shape[-1]) != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), (poses_orig.shape[-1] + poses_virtual.shape[-1])) )
        return
    
    sh = imageio.imread(imgfiles[0]).shape
    poses_orig[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses_orig[2, 4, :] = poses_orig[2, 4, :] * 1./factor
    
    poses_virtual[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses_virtual[2, 4, :] = poses_virtual[2, 4, :] * 1./factor

    # if not load_imgs:
    #     return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
        
    imgs_orig = [imread(f)[...,:3]/255. for f in imgfiles if 'virtual' not in f and not print(f)]
    imgs_orig = np.stack(imgs_orig, -1)  

    imgs_virtual = [imread(f)[...,:3]/255. for f in imgfiles if 'virtual' in f and not print(f)]
    imgs_virtual = np.stack(imgs_virtual, -1) 
    
    print('Loaded ORIGINAL image data', imgs_orig.shape, poses_orig[:,-1,0])
    print('Loaded VIRTUAL image data', imgs_virtual.shape, poses_virtual[:,-1,0])
    
    return poses_orig, bds_orig, imgs_orig, poses_virtual, bds_virtual, imgs_virtual

  
def _load_depth_data(basedir, factor=None, load_depth=True):
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])

    sfx = f'_{factor}'
    _minify(basedir, factor, load_depth=True)
    depth_dir = Path(basedir) / f'depths{sfx}'

    if not os.path.exists(depth_dir):
        print(depth_dir, 'does NOT exist!!!! returning')
        return
    
    depthfiles = [os.path.join(depth_dir, f) for f in sorted(os.listdir(depth_dir)) \
                  if f.lower().endswith('.png')]
    
    if poses.shape[-1] != len(depthfiles):
        print(f'Mismatch between depths {len(depthfiles)} and poses {poses.shape[-1]} !!!!')
        return

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
    
    depths = [imread(f)[..., 0]/255. for f in depthfiles]
    depths = np.stack(depths, -1)
    
    #depths = [iio.imread(f, mode='F') for f in depthfiles]
    print("Loaded depth data:", depths.shape)

    depths = np.moveaxis(depths, -1, 0).astype(np.float32)
    #depths = np.array(depths)
    depths = depths.astype(np.float32)
    print("shape of depths after loading:", depths.shape)

    return depths

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w



def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses
    


def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):
    
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
    
    rays_d = poses[:,:3,2:3]
    rays_o = poses[:,:3,3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    
    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
    
    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc
    
    centroid = np.mean(poses_reset[:,:3,3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2-zh**2)
    new_poses = []
    
    for th in np.linspace(0.,2.*np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
    
    return poses_reset, new_poses, bds
    

def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    

    poses_orig, bds_orig, imgs_orig, poses_virtual, bds_virtual, imgs_virtual = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    #print('Loaded', basedir, bds.min(), bds.max())
    
    # Correct rotation matrix ordering and move variable dim to axis 0
    poses_orig = np.concatenate([poses_orig[:, 1:2, :], -poses_orig[:, 0:1, :], poses_orig[:, 2:, :]], 1)
    poses_orig = np.moveaxis(poses_orig, -1, 0).astype(np.float32)
    imgs_orig = np.moveaxis(imgs_orig, -1, 0).astype(np.float32)
    images_orig = imgs_orig
    bds_orig = np.moveaxis(bds_orig, -1, 0).astype(np.float32)

    poses_virtual = np.concatenate([poses_virtual[:, 1:2, :], -poses_virtual[:, 0:1, :], poses_virtual[:, 2:, :]], 1)
    poses_virtual = np.moveaxis(poses_virtual, -1, 0).astype(np.float32)
    imgs_virtual = np.moveaxis(imgs_virtual, -1, 0).astype(np.float32)
    images_virtual = imgs_virtual
    bds_virtual = np.moveaxis(bds_virtual, -1, 0).astype(np.float32)
    
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds_orig.min() * bd_factor)
    
    poses_orig[:,:3,3] *= sc
    bds_orig *= sc
    
    poses_virtual[:,:3,3] *= sc
    bds_virtual *= sc
    
    poses_all = np.concatenate([poses_orig, poses_virtual], axis=0)

    if recenter:
        # poses_orig = recenter_poses(poses_orig)
        # poses_virtual = recenter_poses(poses_virtual)
        poses_all = recenter_poses(poses_all)
    if spherify:
        poses_orig, render_poses, bds_orig = spherify_poses(poses_orig, bds_orig)
        poses_virtual, _, bds_virtual = spherify_poses(poses_virtual, bds_virtual)

    else:
        
        c2w_orig = poses_avg(poses_orig)
        print('recentered', c2w_orig.shape)
        print(c2w_orig[:3,:4])

        c2w_virtual = poses_avg(poses_virtual)
        print('recentered', c2w_virtual.shape)
        print(c2w_virtual[:3,:4])

        ## Get spiral
        # Get average pose
        up = normalize(poses_orig[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds_orig.min()*.9, bds_orig.max()*5.
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses_orig[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w_orig
        N_views = 120
        N_rots = 2
        if path_zflat:
#             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
        
    n_o = poses_orig.shape[0]
    poses_orig    = poses_all[:n_o]
    poses_virtual = poses_all[n_o:]
        
    render_poses = np.array(render_poses).astype(np.float32)

    c2w_orig = poses_avg(poses_orig)
    print('Data:')
    print(poses_orig.shape, images_orig.shape, bds_orig.shape)
    
    dists = np.sum(np.square(c2w_orig[:3,3] - poses_orig[:,:3,3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)
    
    images_orig = images_orig.astype(np.float32)
    poses_orig = poses_orig.astype(np.float32)

    images_virtual = images_virtual.astype(np.float32)
    poses_virtual = poses_virtual.astype(np.float32)

    return images_orig, poses_orig, bds_orig, images_virtual, poses_virtual, bds_virtual, render_poses, i_test