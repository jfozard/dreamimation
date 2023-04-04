import os
import glob
import tqdm
import math
import imageio
import random
import warnings
import tensorboardX

import numpy as np

import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

import trimesh
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver

import subprocess

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

@torch.cuda.amp.autocast(enabled=False)
def get_rays(ts, poses, intrinsics, H, W, N=-1, error_map=None):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device))
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H*W)

        if error_map is None:
            inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])

    zs = - torch.ones_like(i)
    xs = - (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    # directions = safe_normalize(directions)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)
    rays_d = torch.cat((rays_d, torch.tensor([0.0], device=rays_d.device).expand(B, rays_d.shape[1], 1)), dim=-1)

    rays_o = poses[..., :3, 3] # [B, 3]
    print('rays_o', rays_o.shape)
    rays_o = torch.cat((rays_o, ts.unsqueeze(1).to(rays_o.device)), dim=-1)

    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results



def gradient_x(img: torch.Tensor) -> torch.Tensor:
    return img[:, :, :, :-1] - img[:, :, :, 1:]

def gradient_y(img: torch.Tensor) -> torch.Tensor:
    return img[:, :, :-1, :] - img[:, :, 1:, :]

def depth_smooth_loss(depth):
    grad_x, grad_y = gradient_x(depth), gradient_y(depth)
    return (grad_x.abs().mean() + grad_y.abs().mean()) / 2.

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


def torch_vis_2d(x, renormalize=False):
    # x: [3, H, W] or [1, H, W] or [H, W]
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.permute(1,2,0).squeeze()
        x = x.detach().cpu().numpy()
        
    print(f'[torch_vis_2d] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}')
    
    x = x.astype(np.float32)
    
    # renormalize
    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

    plt.imshow(x)
    plt.show()

@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


class Trainer(object):
    def __init__(self, 
		         argv, # command line args
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 guidance, # guidance network
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):
        
        self.argv = argv
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.frames = opt.frames
        
        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        # guide model
        self.guidance = guidance

        # text prompt
        if self.guidance is not None:
            
            for p in self.guidance.parameters():
                p.requires_grad = False

            self.prepare_text_embeddings()
        
        else:
            self.text_z = None
        
        # try out torch 2.0
        if torch.__version__[0] == '2':
            self.model = torch.compile(self.model)
            self.guidance = torch.compile(self.guidance)
    
        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        print('ema', ema_decay, self.ema)

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)
        
        self.log(f'[INFO] Cmdline: {self.argv}')
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
        self.log(f'[INFO] Options: {self.opt}')
        self.log(f'[INFO] Git hash: {get_git_revision_hash()}')


        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint, allow_rescale=True, model_only=True)

        if opt.static_ckpt:
            self.log(f"[INFO] Loading static {opt.static_ckpt} ...")

            self.load_static(opt.static_ckpt)

    # calculate the text embs.
    def prepare_text_embeddings(self):

        if self.opt.text is None:
            self.log(f"[WARN] text prompt is not provided.")
            self.text_z = None
            return

        if not self.opt.dir_text:
#            self.text_z = None
            self.text_z = self.guidance.get_text_embeds([self.opt.text], [self.opt.negative])
        else:
            self.text_z = []
            for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom']:
                # construct dir-encoded text
                text = f"{self.opt.text}, {d} view"

                negative_text = f"{self.opt.negative}"

                # explicit negative dir-encoded text
                if self.opt.suppress_face:
                    if negative_text != '': negative_text += ', '

                    if d == 'back': negative_text += "face"
                    # elif d == 'front': negative_text += ""
                    elif d == 'side': negative_text += "face"
                    elif d == 'overhead': negative_text += "face"
                    elif d == 'bottom': negative_text += "face"
                
                text_z = self.guidance.get_text_embeds([text], [negative_text])
                self.text_z.append(text_z)
            self.text_z = torch.stack(self.text_z, dim=0)

    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()


    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    ### ------------------------------	

    def train_step(self, ts, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        if self.global_step < self.opt.albedo_iters:
            shading = 'albedo'
            ambient_ratio = 1.0
        else: 
            rand = random.random()
            if rand > 0.8: 
                shading = 'albedo'
                ambient_ratio = 1.0
            elif rand > 0.4: 
                shading = 'textureless'
                ambient_ratio = 0.1
            else: 
                shading = 'lambertian'
                ambient_ratio = 0.1

        bg_color = torch.rand((B * N, 3), device=rays_o.device) # pixel-wise random
        outputs = self.model.render(ts, rays_o, rays_d, staged=False, perturb=True, bg_color=bg_color, ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True, **vars(self.opt))
        pred_rgb = outputs['image'].reshape(B, H, W, 4).permute(0, 3, 1, 2).contiguous() # [1, 3, H, W]
        pred_depth = outputs['depth'].reshape(B, 1, H, W)
        
        # torch_vis_2d(pred_rgb[0])
        
        # text embeddings
        if self.opt.dir_text:
            dirs = data['dir'] # [B,]
            text_z = self.text_z[dirs]
        else:
            text_z = self.text_z
        
        
        # encode pred_rgb to latents

        print('pred_rgb', pred_rgb.shape)
        if self.guidance:
            loss = self.guidance.train_step(text_z, pred_rgb, guidance_scale=self.opt.guide_scale)
        else:
            loss = 0.0
            
        # regularizations
        if self.opt.lambda_opacity > 0:
            loss_opacity = (outputs['weights_sum'] ** 2).mean()
            loss = loss + self.opt.lambda_opacity * loss_opacity

        if self.opt.lambda_stretch>0:
            loss_stretch = outputs['stretch'].mean()
            loss = loss + self.opt.lambda_stretch * loss_stretch


        if self.opt.lambda_entropy > 0:
            alphas = outputs['weights_sum'].clamp(1e-5, 1 - 1e-5)
            # alphas = alphas ** 2 # skewed entropy, favors 0 over 1
            loss_entropy = (- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).mean()
                    
            loss = loss + self.opt.lambda_entropy * loss_entropy

        if self.opt.lambda_orient > 0 and 'loss_orient' in outputs:
            loss_orient = outputs['loss_orient']
            loss = loss + self.opt.lambda_orient * loss_orient
        #loss = torch.tensor([1.0]).cuda()

        if self.opt.lambda_tv >0:
            loss = loss + self.opt.lambda_tv*self.model.loss_tv()

        if self.opt.lambda_emptiness>0:
            loss = loss + self.opt.lambda_emptiness*outputs['scaled_emptiness'].mean()

        if self.opt.lambda_clip >0:
            loss = loss + self.opt.lambda_clip*(torch.exp(pred_rgb**2-self.opt.lambda_clip_r**2).mean())

        if self.opt.lambda_depth>0:
            loss = loss + self.opt.lambda_depth*depth_smooth_loss(pred_depth)
            
        return pred_rgb, pred_depth, loss
    
    def post_train_step(self):

        if self.opt.backbone == 'grid' and self.opt.lambda_tv > 0:

            lambda_tv = min(1.0, self.global_step / 1000) * self.opt.lambda_tv
            # unscale grad before modifying it!
            # ref: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
            self.scaler.unscale_(self.optimizer)
            self.model.encoder.grad_total_variation(lambda_tv, None, self.model.bound)

    def eval_step(self, ts, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]


        
        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        print('eval size', B, N, H, W)
        
        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None

        outputs = self.model.render(ts, rays_o, rays_d, staged=True, perturb=True, bg_color=None, light_d=light_d, ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True, **vars(self.opt))

        print('eval H W', H, W)
        
        pred_rgb = outputs['image'].reshape(B, H, W, 4)
        #print('shapes', outputs['image'].shape, pred_rgb.shape)
        pred_depth = F.interpolate(outputs['depth'].reshape(B, 1, H, W), size=(H*8, W*8), mode='bilinear', align_corners=False).squeeze(1)
        pred_mask = F.interpolate(outputs['weights_sum'].reshape(B, 1, H, W), size=(H*8, W*8), mode='bilinear', align_corners=False).squeeze(1)

        # dummy 
        loss = torch.zeros([1], device=pred_rgb.device, dtype=pred_rgb.dtype)

        return pred_rgb, pred_depth, pred_mask, loss

    def test_step(self, data, bg_color=None, perturb=True):  
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(rays_o.device)
        else:
            bg_color = torch.ones(3, device=rays_o.device) # [3]

        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None

        outputs = self.model.render(rays_o, rays_d, staged=True, perturb=perturb, light_d=light_d, ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True, bg_color=bg_color, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)
        pred_mask = outputs['weights_sum'].reshape(B, H, W) > 0.95

        return pred_rgb, pred_depth, pred_mask

    def generate_point_cloud(self, loader):

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        all_points = []
        all_normals = []

        with torch.no_grad():

            for i, data in enumerate(loader):

                data['shading'] = 'normal' # to get normal as color
                
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, preds_mask = self.test_step(data)

                pred_mask = preds_mask[0].detach().cpu().numpy().reshape(-1) # [H, W], bool
                pred_depth = preds_depth[0].detach().cpu().numpy().reshape(-1, 1) # [N, 1]

                normals = preds[0].detach().cpu().numpy() * 2 - 1 # normals in [-1, 1]
                normals = normals.reshape(-1, 3) # shape [N, 3]

                rays_o = data['rays_o'][0].detach().cpu().numpy() # [N, 3]
                rays_d = data['rays_d'][0].detach().cpu().numpy() # [N, 3]
                points = rays_o + pred_depth * rays_d

                if pred_mask.any():
                    all_points.append(points[pred_mask])
                    all_normals.append(normals[pred_mask])

                pbar.update(loader.batch_size)
        
        points = np.concatenate(all_points, axis=0)
        normals = np.concatenate(all_normals, axis=0)
            
        return points, normals


    def save_mesh(self, loader=None, save_path=None):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'mesh')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(save_path, exist_ok=True)

        if loader is None: # mcubes
            self.model.export_mesh(save_path, decimate_target=self.opt.decimate_target, resolution=self.opt.mcubes_resolution, bbox_size=self.opt.mcubes_bbox)
        else: # poisson (TODO: not working currently...)
            points, normals = self.generate_point_cloud(loader)
            self.model.export_mesh(save_path, points=points, normals=normals, decimate_target=self.opt.decimate_target, resolution=self.opt.mcubes_resolution, bbox_size=self.opt.mcubes_bbox)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):

#        assert self.text_z is not None, 'Training must provide a text prompt!'

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        start_t = time.time()
        
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

        end_t = time.time()

        self.log(f"[INFO] training takes {(end_t - start_t)/ 60:.4f} minutes.")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def render(self, loader, save_path=None, name=None, write_video=True):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
        k = 0
        with torch.no_grad():
            
            for i, data in enumerate(loader):
                
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, preds_mask, _ = self.eval_step(self.global_step, data)
                    print('preds', preds.shape)
                    pred_imgs = []
                    #preds = self.guidance.decode_latents(preds.permute(0,3,1,2)).permute(0,2,3,1)

                    for j in range(preds.shape[0]):
                        pred = F.interpolate(preds[j:j+1].permute(0,3,1,2), (self.opt.dwh, self.opt.dwh),  mode='bilinear', antialias=True)              
                        #print(pred.shape)
                        #pred = preds[j:j+1].permute(0,3,1,2)
                        pred = self.guidance.decode_latents(pred).permute(0,2,3,1)
                        #pred = preds[j:j+1]
                        pred = pred.detach().cpu().numpy()
                        pred = (pred * 255).astype(np.uint8)

                        #print(pred.shape)
                        pred_imgs.append(pred)



                pred_imgs = np.concatenate(pred_imgs, axis=0)

                if write_video:
                    all_preds.append(pred_imgs)
                
                for j in range(pred_imgs.shape[0]):
                    cv2.imwrite(os.path.join(save_path, f'{name}_{k:04d}_rgb.png'), cv2.cvtColor(pred_imgs[j], cv2.COLOR_RGB2BGR))
                    k+=1
                pbar.update(loader.batch_size)

        if write_video:
            all_preds = np.concatenate(all_preds, axis=0)            
            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)

        self.log(f"==> Finished Test.")
    

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training {self.workspace} Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            
            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
                    
            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                pred_rgbs, pred_depths, loss = self.train_step(self.global_step, data)
         
            self.scaler.scale(loss).backward()
            self.post_train_step()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                # if self.report_metric_at_train:
                #     for metric in self.metrics:
                #         metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")


    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate {self.workspace} at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            print('get data')
            for data in iter(loader):

                self.local_step += 1

                print(data)
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, preds_mask, loss = self.eval_step(self.global_step, data)
                    print('latents range', preds.amin(dim=(0,-1,-2)), preds.mean(dim=(0,-1,-2)), preds.amax(dim=(0,-1,-2)))
                    preds = self.guidance.decode_latents(preds.permute(0,3,1,2)).permute(0,2,3,1)

                print('eval_preds', preds.shape)
                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    preds_mask_list = [torch.zeros_like(preds_mask).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_mask_list, preds_mask)
                    preds_mask = torch.cat(preds_mask_list, dim=0)
                
                loss_val = loss.item()
                total_loss += loss_val

                print('eval_preds', preds.shape)

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:
                        
                    save_path = os.path.join(self.workspace, 'validation')
                    save_path_depth = os.path.join(self.workspace, 'validation_depth')
                    save_path_mask = os.path.join(self.workspace, 'validation_mask')
                    name = f'{name}_{self.local_step:04d}_'

                    self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(save_path, exist_ok=True)
                    os.makedirs(save_path_depth, exist_ok=True)
                    os.makedirs(save_path_mask, exist_ok=True)
                    
                    pred = preds.detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                    pred_depth = preds_depth.detach().cpu().numpy()
                    pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-6)
                    pred_depth = (pred_depth * 255).astype(np.uint8)
                  
                    pred_mask = preds_mask.detach().cpu().numpy()
                    pred_mask = (pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min() + 1e-6)
                    pred_mask = (pred_mask * 255).astype(np.uint8)
                  
                    # save image
                    for j in range(preds.shape[0]):

                        print('save', os.path.join(save_path, name+f'{j:03d}_rgb.png'))
                        cv2.imwrite(os.path.join(save_path, name+f'{j:03d}_rgb.png'), cv2.cvtColor(pred[j], cv2.COLOR_RGB2BGR))
                        cv2.imwrite(os.path.join(save_path_depth, name+f'{j:03d}.png'), pred_depth[j])
                        cv2.imwrite(os.path.join(save_path_mask, name+f'{j:03d}.png'), pred_mask[j])

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)


        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{name}.pth"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = os.path.join(self.ckpt_path, self.stats["checkpoints"].pop(0))
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, os.path.join(self.ckpt_path, file_path))

        else:    
            if len(self.stats["results"]) > 0:
                # always save best since loss cannot reflect performance.
                if True:
                    # self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    # self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None, model_only=False, allow_rescale=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        

        if 'model' not in checkpoint_dict:
            if allow_rescale:
                self.model.load_state_dict_rescale(checkpoint_dict)
            else:
               self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        if allow_rescale:
            missing_keys, unexpected_keys = self.model.load_state_dict_rescale(checkpoint_dict['model'], strict=False)
        else:
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        """
        if self.ema is not None and 'ema' in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")
        """
                
        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")
        
        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")
        
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")
        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")


    def load_static(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")

                return
        """
        def expand(ckpt):
            expanded = []
            for k,v in ckpt.items():
                if '_vec' in k:
                    v = v.expand(-1,-1,-1,16)
                    expanded.append((k,v))
                if 'aabb' in k:
                    aabb = torch.cat([v[:3],torch.tensor([0], device=v.device, dtype=v.dtype),v[3:], torch.tensor([1], device=v.device, dtype=v.dtype)])
                    expanded.append((k, aabb))
            ckpt.update(expanded)
            return ckpt

        """
            
        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.static.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        print('chkpt keys', checkpoint_dict['model'].keys())

        missing_keys, unexpected_keys = self.model.static.load_state_dict((checkpoint_dict['model']), strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] static missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        if self.ema is not None and 'ema' in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")



