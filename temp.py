def train():

    parser = config_parser()
    args = parser.parse_args()
    """
    if args.mps:
        device = torch.device(f"mps" if torch.backends.mps.is_available() else "cpu")
        print("****************** MAC POWER ******************")
    """

    print(f"Device is: *********{device}*********")

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images_orig, poses_orig, bds, images_virtual, poses_virtual, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        num_orig = images_orig.shape[0]
        num_virtual=images_virtual.shape[0]
        print('Number of original images: ', num_orig)
        print(images_orig.shape, poses_orig.shape)

        #images_virtual = images_virtual[:num_orig, ]
        #poses_virtual = poses_virtual[:num_orig, ]

        if args.depth_supervision:
            depths = _load_depth_data(args.datadir, args.factor, load_depth=True) 
            print(f'depth shape: {depths.shape}') # numpy array

        hwf = poses_orig[0,:3,-1]
        poses_orig = poses_orig[:,:3,:4]
        poses_virtual = poses_virtual[:,:3,:4]
        print("Virtual poses shape:----------> ", poses_virtual.shape)
        #print(f'Loaded llff, images shape: {images.shape}, render poses shape: {render_poses.shape}, hwf: {hwf}, data dir: {args.datadir}')
        
        if args.i_test is not None:
            i_test = [args.i_test]

        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images_orig.shape[0])[::args.llffhold]

        total_num = images_orig.shape[0] + images_virtual.shape[0]
        i_val = i_test
        i_train = np.array([i for i in np.arange(total_num) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
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
    
    if args.render_only: # To be done later!
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

    N_rand_orig = int(args.N_rand * (num_orig/total_num))
    print('num_orig: ', num_orig,"num_virtual",num_virtual, "total_num: ", total_num, "N_rand: ",args.N_rand,"num_rand_orig:", N_rand_orig)
    N_rand_virtual = args.N_rand - N_rand_orig
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        print("****************---************ Poses orig: ", poses_orig)
        print("****************---************ Poses virtual: ", poses_virtual)
        rays_orig = np.stack([get_rays_np(H, W, K, p) for p in poses_orig[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        rays_virtual = np.stack([get_rays_np(H, W, K, p) for p in poses_virtual[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]

        # Rays shape:   [N, ro+rd, H, W, 3] --> Rays_orig shape: ?? // Rays_virtual shape: ??
        # Images shape: [N, H, W, 3] --> Images_orig shape: ?? // Images_virtual shape: ??
        # Depths shape: [N, H, W]
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
            rays_rgb_orig = np.stack([rays_rgb_orig[i] for i in i_train[0:num_orig-2]], 0) # train images only, Needs to be corrected and generalized later!
            rays_rgb_orig = np.reshape(rays_rgb_orig, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
            rays_rgb_orig = rays_rgb_orig.astype(np.float32)
            np.random.shuffle(rays_rgb_orig)

            rays_rgb_virtual = np.concatenate([rays_virtual, images_virtual[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
            rays_rgb_virtual = np.transpose(rays_rgb_virtual, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
            print("---------*********************", rays_rgb_virtual.shape)
            print(i_train[num_orig-1:total_num-1].shape)
            rays_rgb_virtual = np.stack([rays_rgb_virtual[i] for i in range(9)], 0) # train images only
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

    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            if not args.depth_supervision:
            #     # Random over all images
            #     batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            #     batch = torch.transpose(batch, 0, 1)
            #     batch_rays, target_s = batch[:2], batch[2]

            #     i_batch += N_rand
            #     if i_batch >= rays_rgb.shape[0]:
            #         print("Shuffle data after an epoch!")
            #         rand_idx = torch.randperm(rays_rgb.shape[0])
            #         rays_rgb = rays_rgb[rand_idx]
            #         i_batch = 0

            # ---- SAMPLE BATCH FROM ORIGINAL ----
                if (i_orig + N_rand_orig)<=rays_rgb_orig.shape[0]:
                    batch_orig = rays_rgb_orig[i_orig:i_orig+N_rand_orig]
                    batch_orig= torch.transpose(batch_orig, 0, 1)
                    batch_rays_orig, target_orig = batch_orig[:2], batch_orig[2]
                    #print("orig", i_orig, i_orig+N_rand_orig)
                    i_orig = (i_orig + N_rand_orig)
                else: 
                    batch_orig = rays_rgb_orig[i_orig:i_orig+N_rand_orig]
                    batch_orig= torch.transpose(batch_orig, 0, 1)
                    batch_rays_orig, target_orig = batch_orig[:2], batch_orig[2]

                # ---- SAMPLE BATCH FROM AUGMENTED ----
                batch_virtual = rays_rgb_virtual[i_virtual:i_virtual+N_rand_virtual]
                batch_virtual= torch.transpose(batch_virtual, 0, 1)
                batch_rays_virtual, target_virtual = batch_virtual[:2], batch_virtual[2]
                #print("\nvirtual", i_virtual, i_virtual+N_rand_virtual)
                i_virtual = (i_virtual + N_rand_virtual)


                if i_virtual >= rays_rgb_virtual.shape[0]:
                     print("Shuffle data after an epoch!")
                     rand_idx = torch.randperm(rays_rgb_orig.shape[0])
                     rays_rgb_orig = rays_rgb_orig[rand_idx]
                     rays_rgb_virtual = rays_rgb_virtual[rand_idx]
                     i_orig = 0
                     i_virtual=0
            else:
                if use_batching:
                    # Random over all images
                    batch = rays_rgbd[i_batch:i_batch+N_rand]  # [B, 4, 3]
                    batch = torch.transpose(batch, 0, 1)
                    
                    # Split into components:
                    # batch_rays: origin (3) + direction (3) [2, B, 3]
                    # target_s: RGB (3) + depth (3) [2, B, 3]
                    batch_rays, target_s = batch[:2], batch[2:]
                    
                    # For NeRF compatibility, we might want to separate RGB and depth
                    target_rgb = target_s[0]  # [B, 3] - RGB values
                    #print("---------------********************------------------", target_rgb)

                    target_d = target_s[1][:, 0:1]  # [B, 1] - Take only first channel (repeated depth)
                    #print("---------------********************------------------", target_d)
                    i_batch += N_rand
                    if i_batch >= rays_rgbd.shape[0]:
                        print("Shuffle data after an epoch!")
                        rand_idx = torch.randperm(rays_rgbd.shape[0])
                        rays_rgbd = rays_rgbd[rand_idx]
                        i_batch = 0
        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose).to(device))  # (H, W, 3), (H, W, 3)

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

        #####  Core optimization loop  #####
        rgb_orig, disp, acc, depth, extras_orig = render(H, W, K, chunk=args.chunk, rays=batch_rays_orig,network_query_fn=render_kwargs_train['network_fn_orig'],head='orig',
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)
        rgb_virtual, disp, acc, depth, extras_virtual = render(H, W, K, chunk=args.chunk, rays=batch_rays_virtual,network_query_fn=render_kwargs_train['network_fn_virt'],head='virt',
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)
        
        optimizer.zero_grad()
        if not args.depth_supervision:
            img_loss_orig = img2mse(rgb_orig, target_orig)
            img_loss_virtual = img2mse(rgb_virtual, target_virtual)
            trans = extras_orig['raw'][...,-1]
            loss = img_loss_orig+args.landa*img_loss_virtual
            psnr = mse2psnr(loss)

            if 'rgb0' in extras_orig:
                img_loss0_orig = img2mse(extras_orig['rgb0'], target_orig)
                img_loss0_virtual = img2mse(extras_virtual['rgb0'], target_virtual)

                loss = loss + img_loss0_orig+img_loss0_virtual
                psnr0_orig= mse2psnr(img_loss0_orig)
                psnr0_virtual= mse2psnr(img_loss0_virtual)



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
                rgbs, disps, depths = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'depth.mp4', to8b(depths / np.max(depths)), fps=30, quality=8)
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
            '''
            with torch.no_grad():
                # Get a sample test image
                test_img_idx = 0  # or choose another index
                test_target = images[i_test][test_img_idx]
                test_pose = poses[i_test][test_img_idx]
                test_rays_o, test_rays_d = get_rays(H, W, K, torch.Tensor(test_pose))
                test_batch_rays = torch.stack([test_rays_o, test_rays_d], 0)
                test_rgb, _, _, _, _ = render(H, W, K, chunk=args.chunk, rays=test_batch_rays,
                                            **render_kwargs_test)
                
                # Convert to numpy and log
                test_rgb_np = test_rgb.cpu().numpy().reshape(H, W, 3)
                writer.add_image('Test/rendered', test_rgb_np, i, dataformats='HWC')
                writer.add_image('Test/ground_truth', test_target, i, dataformats='HWC')
            '''
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
        #writer.close()


if __name__=='__main__':
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # Set default tensor type based on available device
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    elif torch.backends.mps.is_available():
        # MPS doesn't have a specific tensor type, just use normal tensors
        torch.set_default_dtype(torch.float32)
    else:
        torch.set_default_dtype(torch.float32)
    
    train()