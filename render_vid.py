import torch
import argparse

from nerf.vid_provider import VidNeRFDataset
from nerf.utils_video import *

from nerf.gui import NeRFGUI

import sys

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--res', default=128, help="static resolution")
    parser.add_argument('--d_res', default=8, help="static resolution")
    parser.add_argument('--d_frames', type=int, default=16, help="num frames for NeRF in training")

    parser.add_argument('--static_ckpt', default=None, help="static initial checkpoint")

    parser.add_argument('--text', default='A corgi dancing, slow motion, smooth', help="text prompt")
    parser.add_argument('--negative', default='low quality, jerky', type=str, help="negative text prompt")

    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --dir_text")
    parser.add_argument('-O2', action='store_true', help="equals --backbone vanilla --dir_text")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--eval_interval', type=int, default=1, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--guidance', type=str, default='stable-diffusion', help='choose from [stable-diffusion, clip]')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--save_mesh', action='store_true', help="export an obj mesh with texture")
    parser.add_argument('--mcubes_resolution', type=int, default=256, help="mcubes resolution for extracting mesh")
    parser.add_argument('--decimate_target', type=int, default=1e5, help="target face number for mesh decimation")

    ### training options
    parser.add_argument('--iters', type=int, default=100000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-3, help="max learning rate")
    parser.add_argument('--warm_iters', type=int, default=500, help="training iters")
    parser.add_argument('--min_lr', type=float, default=1e-4, help="minimal learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=512, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=128, help="num steps sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=128, help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=2048, help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
    parser.add_argument('--albedo', action='store_true', help="only use albedo shading to train, overrides --albedo_iters")
    parser.add_argument('--albedo_iters', type=int, default=1000, help="training iters that only use albedo shading")
    parser.add_argument('--uniform_sphere_rate', type=float, default=0.5, help="likelihood of sampling camera location uniformly on the sphere surface area")
    # model options
    parser.add_argument('--bg_radius', type=float, default=2.5, help="if positive, use a background model at sphere(bg_radius)")
    parser.add_argument('--density_activation', type=str, default='softplus', choices=['softplus', 'exp'], help="density activation function")
    parser.add_argument('--density_thresh', type=float, default=0.1, help="threshold for density grid to be occupied")
    parser.add_argument('--blob_density', type=float, default=5, help="max (center) density for the density blob")
    parser.add_argument('--blob_radius', type=float, default=0.4, help="control the radius for the density blob")
    # network backbone
    parser.add_argument('--fp16', default=True, action='store_true', help="use amp mixed precision training")
    parser.add_argument('--backbone', type=str, default='tensoRF', choices=['grid', 'vanilla', 'tensoRF'], help="nerf backbone")
    parser.add_argument('--optim', type=str, default='adam', choices=['adan', 'adam', 'adamw'], help="optimizer")
    parser.add_argument('--sd_version', type=str, default='1.5', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    # rendering resolution in training, decrease this if CUDA OOM.
    parser.add_argument('--w', type=int, default=32, help="render width for NeRF in training")
    parser.add_argument('--h', type=int, default=32, help="render height for NeRF in training")
    parser.add_argument('--frames', type=int, default=64, help="num frames for NeRF in training")
    parser.add_argument('--jitter_pose', action='store_true', help="add jitters to the randomly sampled camera poses")
    
    ### dataset options
    parser.add_argument('--bound', type=float, default=1.0, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--dt_gamma', type=float, default=0, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.1, help="minimum near distance for camera")
    parser.add_argument('--radius_range', type=float, nargs='*', default=[1.7, 1.8], help="training camera radius range")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[50, 50], help="training camera fovy range")
    parser.add_argument('--dir_text', action='store_true', help="direction-encode the text prompt, by appending front/side/back/overhead view")
    parser.add_argument('--suppress_face', action='store_true', help="also use negative dir text prompt.")
    parser.add_argument('--angle_overhead', type=float, default=30, help="[0, angle_overhead] is the overhead region")
    parser.add_argument('--angle_front', type=float, default=60, help="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")

    parser.add_argument('--lambda_entropy', type=float, default=1e-3, help="loss scale for alpha entropy") ## was 1e-3
    parser.add_argument('--lambda_opacity', type=float, default=0, help="loss scale for alpha value")
    parser.add_argument('--lambda_orient', type=float, default=0e-2, help="loss scale for orientation") ## was 1e-2
    parser.add_argument('--lambda_tv', type=float, default=1e-7, help="loss scale for total variation")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=32*8, help="render width")
    parser.add_argument('--H', type=int, default=32*8, help="render height")
    parser.add_argument('--dwh', type=int, default=64, help="render latents h/w")

    parser.add_argument('--radius', type=float, default=3, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=60, help="default GUI camera fovy")
    parser.add_argument('--light_theta', type=float, default=60, help="default GUI light direction in [0, 180], corresponding to elevation [90, -90]")
    parser.add_argument('--light_phi', type=float, default=0, help="default GUI light direction in [0, 360), azimuth")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

    opt = parser.parse_args()

    if opt.O:
        opt.fp16 = True
        opt.dir_text = True
        opt.cuda_ray = True

    elif opt.O2:
        # only use fp16 if not evaluating normals (else lead to NaNs in training...)
        if opt.albedo:
            opt.fp16 = True
        opt.dir_text = True
        opt.backbone = 'vanilla'

    if opt.albedo:
        opt.albedo_iters = opt.iters

    if opt.backbone == 'vanilla':
        from nerf.network import NeRFNetwork
    elif opt.backbone == 'grid':
        from nerf.network_grid import NeRFNetwork
    elif opt.backbone == 'tensoRF':
        from nerf.network_tensoRF_deform import DeformNeRFNetwork
    else:
        raise NotImplementedError(f'--backbone {opt.backbone} is not implemented!')

    seed_everything(opt.seed)

    model = DeformNeRFNetwork(opt)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.guidance == 'stable-diffusion':
        from sd_video import VideoDiffusion #TextToVideoSynthesis
        guidance = VideoDiffusion(device)
    elif opt.guidance == 'clip':
        from nerf.clip import CLIP
        guidance = CLIP(device)
    else:
        raise NotImplementedError(f'--guidance {opt.guidance} is not implemented.')

    trainer = Trainer(' '.join(sys.argv), 'df', opt, model, guidance, device=device, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint=opt.ckpt)

    render_loader = VidNeRFDataset(opt, device=device, type='render', H=opt.H, W=opt.W, size=opt.frames, frames=opt.frames).dataloader()
    trainer.render(render_loader)
