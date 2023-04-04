import torch
import torch.nn as nn
import torch.nn.functional as F


from activation import trunc_exp

from .renderer import NeRFRenderer

import numpy as np
from encoding import get_encoder

from .utils import safe_normalize

import numpy as np

import itertools

def repeat_el(lst, n):
    return list(itertools.chain.from_iterable(itertools.repeat(x, n) for x in lst))

class BasicBlock(nn.Module):
    def __init__(self, dim_in, dim_out, activation='softplus', norm=False, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.norm = nn.LayerNorm(dim_out) if norm else nn.Identity()

        self.dense = nn.Linear(self.dim_in, self.dim_out, bias=bias)
        if norm:
            with torch.no_grad():
                nn.init.zeros_(self.dense.bias)

        if activation == 'softplus':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [B, C]

        out = self.dense(x)
        out = self.norm(out)
        out = self.activation(out)

        return out    

    
    
class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, activation='softplus', bias=True, bias_out=True, block=BasicBlock, norm=False):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []

        if num_layers==1:
            l = nn.Linear(self.dim_in, self.dim_out, bias=bias_out)
            if norm and bias_out:
                with torch.no_grad():
                    nn.init.zeros_(l.bias)
            net.append(l)
        else:            
            for l in range(num_layers):
                if l == 0:
                    net.append(BasicBlock(self.dim_in, self.dim_hidden, activation=activation, norm=norm, bias=bias))
                elif l != num_layers - 1:
                    net.append(block(self.dim_hidden, self.dim_hidden, activation=activation, norm=norm, bias=bias))
                else:
                    l = nn.Linear(self.dim_hidden, self.dim_out, bias=bias_out)
                    if norm and bias_out:
                        with torch.no_grad():
                            nn.init.zeros_(l.bias)
                    net.append(l)
                
            
        self.net = nn.ModuleList(net)
        
    
    def forward(self, x):

        for l in range(self.num_layers):
            x = self.net[l](x)
            
        return x

class DeformNeRFNetwork(NeRFRenderer):
    def __init__(self,
                 opt,
                 n_resolutions=1,
                 resolution=[[8], [8], [8], [16]],
                 d_rank= [4],
                 num_layers=1, #3,
                 hidden_dim=8, #128,

                 ):
        super().__init__(opt)

        self.static = NeRFNetwork(opt)

        #for name, param in self.static.named_parameters():
        #    if 'bg' not in name:
        #        param.requires_grad = False

        self.resolution = [[opt.d_res]]*3 + [[opt.d_frames]]

        self.n_resolutions = n_resolutions

        print('resolution', resolution)
        
        # vector-matrix decomposition
        self.d_rank = d_rank


        self.mat_ids = [[0, 1], [0, 2], [1, 2]]
        self.vec_ids = [[3, 2], [3, 1], [3, 0]]

        self.mat_ids_rep = repeat_el(self.mat_ids, n_resolutions) 

        self.vec_ids_rep = repeat_el(self.vec_ids, n_resolutions) 

        self.d_mat, self.d_vec = self.init_one_svd(self.d_rank, self.resolution)
        self.d_net = MLP(sum(self.d_rank)*3, 3, hidden_dim, num_layers, bias=True)

        self.d_scale = opt.d_scale

        self.opt = opt

    def init_one_svd(self, n_component, resolution, scale=(1.0, 1.0)):

        mat = []
        vec = []

        for i in range(len(self.mat_ids)):
            mat_id_0, mat_id_1 = self.mat_ids[i]
            vec_id_0, vec_id_1 = self.vec_ids[i]
            for j in range(len(resolution[mat_id_0])):
               # print(n_component, i, resolution[mat_id_0], vec_id, j)

                if j==0:
                    mat.append(nn.Parameter(scale[0] * torch.randn((1, n_component[j], resolution[mat_id_1][j], resolution[mat_id_0][j])))) # [1, R, H, W]
                    vec.append(nn.Parameter(scale[0] * torch.randn((1, n_component[j], resolution[vec_id_1][j], resolution[vec_id_0][j])))) # [1, R, F, W]
                else:
                    mat.append(nn.Parameter(scale[1] * torch.randn((1, n_component[j], resolution[mat_id_1][j], resolution[mat_id_0][j])))) # [1, R, H, W]
                    vec.append(nn.Parameter(scale[1] * torch.randn((1, n_component[j], resolution[vec_id_1][j], resolution[vec_id_0][j])))) # [1, R, F, W]

        print('matrix sizes', [m.shape for m in mat])
        print('vector sizes', [v.shape for v in vec])
        return nn.ParameterList(mat), nn.ParameterList(vec)



    def deform(self, x):
        # x: [N, 4], in [-1, 1]

        N = x.shape[0]


        mat_feat = []
        vec_feat = []

        for j in range(3):
            for i in range(j*self.n_resolutions, (j+1)*self.n_resolutions):
                mat_coord = x[..., self.mat_ids_rep[i]].view(1,-1,1,2)
                mat_feat.append(F.grid_sample(self.d_mat[i], mat_coord, align_corners=True).view(-1, N)) # [1, R, N, 1] --> [R, N]
                vec_coord = x[ ..., self.vec_ids_rep[i]].view(1,-1,1,2)
                vec_feat.append(F.grid_sample(self.d_vec[i], vec_coord, align_corners=True).view(-1, N))

        mat_feat = torch.cat(mat_feat, dim=0)
        vec_feat = torch.cat(vec_feat, dim=0)
              
        d_feat = (mat_feat*vec_feat).T
        
        d = self.d_net(d_feat)*self.d_scale

        y = x[:,:3] + d

        return y


    
    def forward(self, ts, x, d, l=None, ratio=1, shading='albedo'):

        x = 2 * (x - self.aabb_train[:4]) / (self.aabb_train[4:] - self.aabb_train[:4]) - 1

        x = self.deform(x)

        return self.static( x, d )

      
    def density(self, ts, x, return_xd=False):
        # x: [N, 3], in [-bound, bound]

        # normalize to [-1, 1] inside aabb_train
        x = 2 * (x - self.aabb_train[:4]) / (self.aabb_train[4:] - self.aabb_train[:4]) - 1

        xd = self.deform(x)

        
        
        res = self.static.density(xd)
        if return_xd:
            res['xd'] = xd

        return res

    
    def background(self, d):

        rgbs = self.static.background(d)

        return rgbs
    
    def loss_tv(self):
        loss = 0.0
        for i in range(3):
            loss += tv2(self.d_mat[i])
            loss += tv2(self.d_vec[i])

        return loss

    def load_state_dict_rescale(self, param_dict, strict=False):


        new_sizes = dict([(p, v.shape) for p, v in self.state_dict().items()])
        print(new_sizes)

        new_param_dict = {}
        for p, v in param_dict.items():
            if v.shape != new_sizes[p]:
                print('RESIZE ', p, v.shape, '->', new_sizes[p])
                v = F.interpolate(v, new_sizes[p][2:], mode='bilinear', antialias=True)
            new_param_dict[p] = v

        return self.load_state_dict(new_param_dict, strict=strict)



    # optimizer utils
    def get_params(self, lr):
        lr1 = lr
        lr2 = lr
        params = [
            {'params': self.d_mat, 'lr': lr1}, 
            {'params': self.d_vec, 'lr': lr1},
            {'params': self.d_net.parameters(), 'lr': lr2},
        ]

        params += self.static.get_params(lr*self.opt.lr_ratio_static)
        #params.append({'params': self.static.bg_net.parameters(), 'lr': lr2 })
        return params
        
def tv2(m):
    B, C, I, J = m.shape
    return ((m[:,:,1:,:] - m[:,:,:-1,:])**2).mean() + ((m[:,:,:,1:] - m[:,:,:,:-1])**2).mean()

    
class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 opt,
                 n_resolutions=1,
                 sigma_rank= [16],
                 color_rank= [16],
                 color_feat_dim=4,
                 num_layers=3, #3,
                 hidden_dim=64, #128,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 ):
        super().__init__(opt)

        self.resolution = [[opt.res]]*3
        self.n_resolutions = n_resolutions
        
        
        # vector-matrix decomposition
        self.sigma_rank = sigma_rank
        self.color_rank = color_rank
        self.color_feat_dim = color_feat_dim

        self.mat_ids = [[0, 1], [0, 2], [1, 2]]
        self.vec_ids = [2, 1, 0]

        self.mat_ids_rep = repeat_el(self.mat_ids, n_resolutions) 

        self.vec_ids_rep = repeat_el(self.vec_ids, n_resolutions) 


        self.sigma_mat, self.sigma_vec = self.init_one_svd(self.sigma_rank, self.resolution)
        self.color_mat, self.color_vec = self.init_one_svd(self.color_rank, self.resolution)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.sigma_net = MLP(sum(self.sigma_rank)*3, 1, hidden_dim, num_layers, bias=True)
        self.color_net = MLP(sum(self.color_rank)*3, 4, hidden_dim, num_layers, bias=True)


            
        self.density_activation = trunc_exp #if self.opt.density_activation == 'exp' else F.softplus


        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg   
            self.hidden_dim_bg = hidden_dim_bg
            
            # use a very simple network to avoid it learning the prompt...
            self.encoder_bg, self.in_dim_bg = get_encoder('frequency', input_dim=3,  multires=4)
            self.bg_net = MLP(self.in_dim_bg, 4, hidden_dim_bg, num_layers_bg, bias=True)
            
        else:
            self.bg_net = None


        aabb_train = torch.FloatTensor([-opt.bound, -opt.bound, -opt.bound, opt.bound, opt.bound, opt.bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

    # add a density blob to the scene center
    def density_blob(self, x):
        # x: [B, N, 3]
        
        d = (x ** 2).sum(-1)
        g = self.opt.blob_density * torch.exp(- d / (self.opt.blob_radius ** 2))

        return g

    def init_one_svd(self, n_component, resolution, scale=(0.1, 0.1)):

        mat = []
        vec = []

        for i in range(len(self.mat_ids)):
            mat_id_0, mat_id_1 = self.mat_ids[i]
            vec_id = self.vec_ids[i]
            for j in range(len(resolution[mat_id_0])):

                if j==0:
                    mat.append(nn.Parameter(scale[0] * torch.randn((1, n_component[j], resolution[mat_id_1][j], resolution[mat_id_0][j])))) # [1, R, H, W]
                    vec.append(nn.Parameter(scale[0] * torch.ones((1, n_component[j], resolution[vec_id][j], 1)))) # [1, R, D, 1] (fake 2d to use grid_sample)
                else:
                    mat.append(nn.Parameter(scale[1] * torch.randn((1, n_component[j], resolution[mat_id_1][j], resolution[mat_id_0][j])))) # [1, R, H, W]
                    vec.append(nn.Parameter(scale[1] * torch.ones((1, n_component[j], resolution[vec_id][j], 1)))) # [1, R, D, 1] (fake 2d to use grid_sample)

            #

        print('matrix sizes', [m.shape for m in mat])
        print('vector sizes', [v.shape for v in vec])
        return nn.ParameterList(mat), nn.ParameterList(vec)


    def get_sigma_feat(self, x):
        # x: [N, 3], in [-1, 1] (outliers will be treated as zero due to grid_sample padding mode)

        N = x.shape[0]

        mat_feat = []
        vec_feat = []
        for j in range(3):
            for i in range(j*self.n_resolutions, (j+1)*self.n_resolutions):
                mat_coord = x[..., self.mat_ids_rep[i]].view(1,-1,1,2)
                mat_feat.append(F.grid_sample(self.sigma_mat[i], mat_coord, align_corners=True).view(-1, N)) # [1, R, N, 1] --> [R, N]
                vec_coord = x[ ..., self.vec_ids_rep[i]].view(1,-1,1)
                vec_coord = torch.stack((torch.zeros_like(vec_coord), vec_coord), dim=-1).view(1, -1, 1, 2) # [1, N, 1, 2], fake 2d coord
                vec_feat.append(F.grid_sample(self.sigma_vec[i], vec_coord, align_corners=True).view(-1, N)) 

        mat_feat = torch.cat(mat_feat, dim=0)
        vec_feat = torch.cat(vec_feat, dim=0)
        sigma_feat = (mat_feat*vec_feat).T

        return sigma_feat


    def get_color_feat(self, x):
        # x: [N, 3], in [-1, 1]

        N = x.shape[0]

        mat_feat = []
        vec_feat = []

        for j in range(3):
            for i in range(j*self.n_resolutions, (j+1)*self.n_resolutions):
                mat_coord = x[..., self.mat_ids_rep[i]].view(1,-1,1,2)
                mat_feat.append(F.grid_sample(self.color_mat[i], mat_coord, align_corners=True).view(-1, N)) # [1, R, N, 1] --> [R, N]
                #vec_coord = torch.stack((x[..., self.vec_ids_rep[0]], x[..., self.vec_ids[1]], x[..., self.vec_ids[2]]))
                vec_coord = x[ ..., self.vec_ids_rep[i]].view(1,-1,1)
                vec_coord = torch.stack((torch.zeros_like(vec_coord), vec_coord), dim=-1).view(1, -1, 1, 2) # [1, N, 1, 2], fake 2d coord
                vec_feat.append(F.grid_sample(self.color_vec[i], vec_coord, align_corners=True).view(-1, N))

        mat_feat = torch.cat(mat_feat, dim=0)
        vec_feat = torch.cat(vec_feat, dim=0)
              
        color_feat = (mat_feat*vec_feat).T
            
        return color_feat

    def get_normal_feat(self, x):
        # x: [N, 3], in [-1, 1]

        N = x.shape[0]

        mat_feat = []
 
        for j in range(3):
            for i in range(j*self.n_resolutions_normal, (j+1)*self.n_resolutions_normal):
                mat_coord = x[..., self.mat_ids_rep_normal[i]].view(1,-1,1,2)
                mat_feat.append(F.grid_sample(self.normal_mat[i], mat_coord, align_corners=True).view(-1, N)) # [1, R, N, 1] --> [R, N]

        normal_feat = torch.cat(mat_feat, axis=0).T

        return normal_feat
   
    
    def forward(self, x, d, l=None, ratio=1, shading='albedo'):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]
        # ratio: scalar, ambient ratio, 1 == no shading (albedo only), 0 == only shading (textureless)

        # normalize to [-1, 1] inside aabb_train
        x = 2 * (x - self.aabb_train[:3]) / (self.aabb_train[3:] - self.aabb_train[:3]) - 1

        # rgb
        color_feat = self.get_color_feat(x)
        color_feat = self.color_net(color_feat)

        
        # sigmoid activation for rgb
        albedo = color_feat#torch.sigmoid(color_feat)

        if shading == 'albedo':
            normal = None
            color = albedo
        
        else: 
            raise NotImplementedError
            
        return color, normal

      
    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        print(x.shape, self.aabb_train.shape)

        # normalize to [-1, 1] inside aabb_train
        x = 2 * (x - self.aabb_train[:3]) / (self.aabb_train[3:] - self.aabb_train[:3]) - 1

        print(x.shape)
        sigma_feat = self.get_sigma_feat(x)
        sigma_feat = self.sigma_net(sigma_feat).squeeze(1)
        blob = self.density_blob(x)        
        sigma = self.density_activation(sigma_feat + blob)

        return {
            'sigma': sigma,
#            'albedo': albedo,
        }


    # ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
    def finite_difference_normal(self, x, epsilon=1e-2):
        # x: [N, 3]
        dx_pos = self.density((x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
        dx_neg = self.density((x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
        dy_pos = self.density((x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
        dy_neg = self.density((x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
        dz_pos = self.density((x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
        dz_neg = self.density((x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
        
        normal = torch.stack([
            0.5 * (dx_pos - dx_neg) / epsilon, 
            0.5 * (dy_pos - dy_neg) / epsilon, 
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        return -normal


    def normal_fd(self, x):

        normal = self.finite_difference_normal(x)
        normal = safe_normalize(normal)
        normal = torch.nan_to_num(normal)

        return normal
    
    def background(self, d):

        h = self.encoder_bg(d) # [N, C]
        
        h = self.bg_net(h)

        # sigmoid activation for rgb
        rgbs = h

        return rgbs

    """
    # L1 penalty for loss
    def density_loss(self):
        loss = 0
        for i in range(len(self.sigma_mat)):
            loss = loss + torch.mean(torch.abs(self.sigma_mat[i])) + torch.mean(torch.abs(self.sigma_vec[i]))
        return loss
    
    # upsample utils
    @torch.no_grad()
    def upsample_params(self, mat, vec, resolution):

        for i in range(len(self.vec_ids)):
            vec_id = self.vec_ids[i]
            mat_id_0, mat_id_1 = self.mat_ids[i]
            mat[i] = nn.Parameter(F.interpolate(mat[i].data, size=(resolution[mat_id_1], resolution[mat_id_0]), mode='bilinear', align_corners=True))
            vec[i] = nn.Parameter(F.interpolate(vec[i].data, size=(resolution[vec_id], 1), mode='bilinear', align_corners=True))


    @torch.no_grad()
    def upsample_model(self, resolution):
        self.upsample_params(self.sigma_mat, self.sigma_vec, resolution)
        self.upsample_params(self.color_mat, self.color_vec, resolution)
        self.resolution = resolution

    @torch.no_grad()
    def shrink_model(self):
        # shrink aabb_train and the model so it only represents the space inside aabb_train.

        half_grid_size = self.bound / self.grid_size
        thresh = min(self.density_thresh, self.mean_density)

        # get new aabb from the coarsest density grid (TODO: from the finest that covers current aabb?)
        valid_grid = self.density_grid[self.cascade - 1] > thresh # [N]
        valid_pos = raymarching.morton3D_invert(torch.nonzero(valid_grid)) # [Nz] --> [Nz, 3], in [0, H - 1]
        #plot_pointcloud(valid_pos.detach().cpu().numpy()) # lots of noisy outliers in hashnerf...
        valid_pos = (2 * valid_pos / (self.grid_size - 1) - 1) * (self.bound - half_grid_size) # [Nz, 3], in [-b+hgs, b-hgs]
        min_pos = valid_pos.amin(0) - half_grid_size # [3]
        max_pos = valid_pos.amax(0) + half_grid_size # [3]

        # shrink model
        reso = torch.LongTensor(self.resolution).to(self.aabb_train.device)
        units = (self.aabb_train[3:] - self.aabb_train[:3]) / reso
        tl = (min_pos - self.aabb_train[:3]) / units
        br = (max_pos - self.aabb_train[:3]) / units
        tl = torch.round(tl).long().clamp(min=0)
        br = torch.minimum(torch.round(br).long(), reso)
        
        for i in range(len(self.vec_ids)):
            vec_id = self.vec_ids[i]
            mat_id_0, mat_id_1 = self.mat_ids[i]

            self.sigma_vec[i] = nn.Parameter(self.sigma_vec[i].data[..., tl[vec_id]:br[vec_id], :])
            self.color_vec[i] = nn.Parameter(self.color_vec[i].data[..., tl[vec_id]:br[vec_id], :])

            self.sigma_mat[i] = nn.Parameter(self.sigma_mat[i].data[..., tl[mat_id_1]:br[mat_id_1], tl[mat_id_0]:br[mat_id_0]])
            self.color_mat[i] = nn.Parameter(self.color_mat[i].data[..., tl[mat_id_1]:br[mat_id_1], tl[mat_id_0]:br[mat_id_0]])
        
        self.aabb_train = torch.cat([min_pos, max_pos], dim=0) # [6]

        print(f'[INFO] shrink slice: {tl.cpu().numpy().tolist()} - {br.cpu().numpy().tolist()}')
        print(f'[INFO] new aabb: {self.aabb_train.cpu().numpy().tolist()}')
    """

    def loss_tv(self):
        loss = 0.0
        for i in range(3):
            loss += tv2(self.sigma_mat[i])
            loss += tv(self.sigma_vec[i])
            loss += tv(self.color_vec[i])
            loss += tv2(self.color_mat[i])
        return loss


    # optimizer utils
    def get_params(self, lr):
        lr1 = lr
        lr2 = lr
        params = [
            {'params': self.sigma_mat, 'lr': lr1}, 
            {'params': self.sigma_vec, 'lr': lr1},

            {'params': self.color_mat, 'lr': lr1}, 
            {'params': self.color_vec, 'lr': lr1},

            {'params': self.sigma_net.parameters(), 'lr': lr2},
            {'params': self.color_net.parameters(), 'lr': lr2},
        ]
        if self.bg_radius > 0:
            params.append({'params': self.bg_net.parameters(), 'lr': lr2 })
        return params
        

def tv(m):
    B, C, I, J = m.shape
    return ((m[:,:,1:,:] - m[:,:,:-1,:])**2).mean()

def tv2(m, t_mult=1.0):
    B, C, I, J = m.shape
    return ((m[:,:,1:,:] - m[:,:,:-1,:])**2).mean() + t_mult*((m[:,:,:,1:] - m[:,:,:,:-1])**2).mean()


