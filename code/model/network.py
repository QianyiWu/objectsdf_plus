import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import rend_util
from model.embedder import *
from model.density import LaplaceDensity
from model.ray_sampler import ErrorBoundSampler
import matplotlib.pyplot as plt
import numpy as np

from torch import vmap

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            sdf_bounding_sphere,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            sphere_scale=1.0,
            inside_outside=True,
            sigmoid = 10
    ):
        super().__init__()

        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] = input_ch
        print(multires, dims)
        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.d_out = d_out
        self.sigmoid = sigmoid

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    # Geometry initalization for compositional scene, bg SDF sign: inside + outside -, fg SDF sign: outside + inside -
                    # The 0 index is the background SDF, the rest are the object SDFs
                    # background SDF with postive value inside and nagative value outside
                    torch.nn.init.normal_(lin.weight[:1, :], mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias[:1], bias) 
                    # inner objects with SDF initial with negative value inside and positive value outside, ~0.6 radius of background
                    torch.nn.init.normal_(lin.weight[1:,:], mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias[1:], -0.6*bias) 

                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        self.pool = nn.MaxPool1d(self.d_out)

    def forward(self, input):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:self.d_out]
        d_output = torch.ones_like(y[:, :1], requires_grad=False, device=y.device)
        g = []
        for idx in range(y.shape[1]):
            gradients = torch.autograd.grad(
                outputs=y[:, idx:idx+1],
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
            g.append(gradients)
        g = torch.cat(g)
        # add the gradient of minimum sdf
        sdf = -self.pool(-y.unsqueeze(1)).squeeze(-1)
        g_min_sdf = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        g = torch.cat([g, g_min_sdf])
        return g

    def get_outputs(self, x, beta=None):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf_raw = output[:,:self.d_out]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf_raw = torch.minimum(sdf_raw, sphere_sdf.expand(sdf_raw.shape))
        if beta == None:
            semantic = self.sigmoid * torch.sigmoid(-self.sigmoid * sdf_raw)
        else:
            semantic = 0.5/beta *torch.exp(-sdf_raw.abs()/beta)
        sdf = -self.pool(-sdf_raw.unsqueeze(1)).squeeze(-1) # get the minium value of sdf
        feature_vectors = output[:, self.d_out:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients, semantic, sdf_raw

    def get_sdf_vals(self, x):
        sdf = self.forward(x)[:,:self.d_out]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        # sdf = -self.pool(-sdf) # get the minium value of sdf  if bound apply in the final 
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf.expand(sdf.shape))
        sdf = -self.pool(-sdf.unsqueeze(1)).squeeze(-1) # get the minium value of sdf if bound apply before min
        return sdf
    
    def get_sdf_raw(self, x):
        return self.forward(x)[:, :self.d_out]


from hashencoder.hashgrid import HashEncoder
class ObjectImplicitNetworkGrid(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            sdf_bounding_sphere,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0, # radius of the sphere in geometric initialization
            skip_in=(),
            weight_norm=True,
            multires=0,
            sphere_scale=1.0,
            inside_outside=False,
            base_size = 16,
            end_size = 2048,
            logmap = 19,
            num_levels=16,
            level_dim=2,
            divide_factor = 1.5, # used to normalize the points range for multi-res grid
            use_grid_feature = True, # use hash grid embedding or not, if not, it is a pure MLP with sin/cos embedding
            sigmoid = 20
    ):
        super().__init__()
        
        self.d_out = d_out
        self.sigmoid = sigmoid
        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in] + dims + [d_out + feature_vector_size]
        self.embed_fn = None
        self.divide_factor = divide_factor
        self.grid_feature_dim = num_levels * level_dim
        self.use_grid_feature = use_grid_feature
        dims[0] += self.grid_feature_dim
        
        print(f"[INFO]: using hash encoder with {num_levels} levels, each level with feature dim {level_dim}")
        print(f"[INFO]: resolution:{base_size} -> {end_size} with hash map size {logmap}")
        self.encoding = HashEncoder(input_dim=3, num_levels=num_levels, level_dim=level_dim, 
                    per_level_scale=2, base_resolution=base_size, 
                    log2_hashmap_size=logmap, desired_resolution=end_size)
        
        '''
        # can also use tcnn for multi-res grid as it now supports eikonal loss
        base_size = 16
        hash = True
        smoothstep = True
        self.encoding = tcnn.Encoding(3, {
                        "otype": "HashGrid" if hash else "DenseGrid",
                        "n_levels": 16,
                        "n_features_per_level": 2,
                        "log2_hashmap_size": 19,
                        "base_resolution": base_size,
                        "per_level_scale": 1.34,
                        "interpolation": "Smoothstep" if smoothstep else "Linear"
                    })
        '''
        
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] += input_ch - 3
        # print("network architecture")
        # print(dims)
        
        self.num_layers = len(dims)
        self.skip_in = skip_in
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    # Geometry initalization for compositional scene, bg SDF sign: inside + outside -, fg SDF sign: outside + inside -
                    # The 0 index is the background SDF, the rest are the object SDFs
                    # background SDF with postive value inside and nagative value outside
                    torch.nn.init.normal_(lin.weight[:1, :], mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias[:1], bias)
                    # inner objects with SDF initial with negative value inside and positive value outside, ~0.5 radius of background
                    torch.nn.init.normal_(lin.weight[1:,:], mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias[1:], -0.5*bias)

                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        self.cache_sdf = None

        self.pool = nn.MaxPool1d(d_out)
        self.relu = nn.ReLU()

    def forward(self, input):
        if self.use_grid_feature:
            # normalize point range as encoding assume points are in [-1, 1]
            # assert torch.max(input / self.divide_factor)<1 and torch.min(input / self.divide_factor)>-1, 'range out of [-1, 1], max: {}, min: {}'.format(torch.max(input / self.divide_factor),  torch.min(input / self.divide_factor))
            feature = self.encoding(input / self.divide_factor)
        else:
            feature = torch.zeros_like(input[:, :1].repeat(1, self.grid_feature_dim))

        if self.embed_fn is not None:
            embed = self.embed_fn(input)
            input = torch.cat((embed, feature), dim=-1)
        else:
            input = torch.cat((input, feature), dim=-1)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:self.d_out]
        d_output = torch.ones_like(y[:, :1], requires_grad=False, device=y.device)
        f = lambda v: torch.autograd.grad(outputs=y,
                    inputs=x,
                    grad_outputs=v.repeat(y.shape[0], 1),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True)[0]
        
        N = torch.eye(y.shape[1], requires_grad=False).to(y.device)
        
        # start_time = time.time()
        if self.use_grid_feature: # using hashing grid feature, cannot support vmap now
            g = torch.cat([torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=idx.repeat(y.shape[0], 1),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0] for idx in N.unbind()])
        # torch.cuda.synchronize()
        # print("time for computing gradient by for loop: ", time.time() - start_time, "s")
                
        # using vmap for batched gradient computation, if not using grid feature (pure MLP)
        else:
            g = vmap(f, in_dims=1)(N).reshape(-1, 3)
        
        # add the gradient of scene sdf
        sdf = -self.pool(-y.unsqueeze(1)).squeeze(-1)
        g_min_sdf = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        g = torch.cat([g, g_min_sdf])
        return g

    def get_outputs(self, x, beta=None):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf_raw = output[:,:self.d_out]
        # if self.sdf_bounding_sphere > 0.0:
        #     sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
        #     sdf_raw = torch.minimum(sdf_raw, sphere_sdf.expand(sdf_raw.shape))

        if beta == None:
            semantic = self.sigmoid * torch.sigmoid(-self.sigmoid * sdf_raw)
        else:
            # change semantic to the gradianct of density
            semantic = 1/beta * (0.5 + 0.5 * sdf_raw.sign() * torch.expm1(-sdf_raw.abs() / beta))
        sdf = -self.pool(-sdf_raw.unsqueeze(1)).squeeze(-1) # get the minium value of all objects sdf
        feature_vectors = output[:, self.d_out:]

        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients, semantic, sdf_raw

    def get_specific_outputs(self, x, idx):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf_raw = output[:,:self.d_out]
        semantic = self.sigmoid * torch.sigmoid(-self.sigmoid * sdf_raw)
        sdf = -self.pool(-sdf_raw.unsqueeze(1)).squeeze(-1)

        feature_vectors = output[:, self.d_out:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients, semantic, output[:,:self.d_out]


    def get_sdf_vals(self, x):
        sdf = -self.pool(-self.forward(x)[:,:self.d_out].unsqueeze(1)).squeeze(-1)
        return sdf

    def get_sdf_raw(self, x):
        return self.forward(x)[:, :self.d_out]


    def mlp_parameters(self):
        parameters = []
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            parameters += list(lin.parameters())
        return parameters

    def grid_parameters(self, verbose=False):
        if verbose:
            print("[INFO]: grid parameters", len(list(self.encoding.parameters())))
            for p in self.encoding.parameters():
                print(p.shape)
        return self.encoding.parameters()


class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0,
            per_image_code = False
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.per_image_code = per_image_code
        if self.per_image_code:
            # nerf in the wild parameter
            # parameters
            # maximum 1024 images
            self.embeddings = nn.Parameter(torch.empty(1024, 32))
            std = 1e-4
            self.embeddings.data.uniform_(-std, std)
            dims[0] += 32

        # print("rendering network architecture:")
        # print(dims)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, points, normals, view_dirs, feature_vectors, indices):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'nerf':
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)
        else:
            raise NotImplementedError

        if self.per_image_code:
            image_code = self.embeddings[indices].expand(rendering_input.shape[0], -1)
            rendering_input = torch.cat([rendering_input, image_code], dim=-1)
            
        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)
        
        x = self.sigmoid(x)
        return x


class ObjectSDFPlusNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.scene_bounding_sphere = conf.get_float('scene_bounding_sphere', default=1.0)
        self.white_bkgd = conf.get_bool('white_bkgd', default=False)
        self.bg_color = torch.tensor(conf.get_list("bg_color", default=[1.0, 1.0, 1.0])).float().cuda()

        Grid_MLP = conf.get_bool('Grid_MLP', default=False)
        self.Grid_MLP = Grid_MLP
        if Grid_MLP: 
            self.implicit_network = ObjectImplicitNetworkGrid(self.feature_vector_size, 0.0 if self.white_bkgd else self.scene_bounding_sphere, **conf.get_config('implicit_network'))    
        else:
            self.implicit_network = ImplicitNetwork(self.feature_vector_size, 0.0 if self.white_bkgd else self.scene_bounding_sphere, **conf.get_config('implicit_network'))
        
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        
        self.density = LaplaceDensity(**conf.get_config('density'))
        self.ray_sampler = ErrorBoundSampler(self.scene_bounding_sphere, **conf.get_config('ray_sampler'))
        
        self.num_semantic = conf.get_int('implicit_network.d_out')

    def forward(self, input, indices):
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        
        # we should use unnormalized ray direction for depth
        ray_dirs_tmp, _ = rend_util.get_camera_params(uv, torch.eye(4).to(pose.device)[None], intrinsics)
        depth_scale = ray_dirs_tmp[0, :, 2:]
        
        batch_size, num_pixels, _ = ray_dirs.shape

        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self)
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1,N_samples,1)
        dirs_flat = dirs.reshape(-1, 3)

        beta_cur = self.density.get_beta()
        sdf, feature_vectors, gradients, semantic, sdf_raw = self.implicit_network.get_outputs(points_flat, beta=None)

        rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors, indices)
        rgb = rgb_flat.reshape(-1, N_samples, 3)

        semantic = semantic.reshape(-1, N_samples, self.num_semantic)
        weights, transmittance, dists = self.volume_rendering(z_vals, sdf) 

        # rendering the occlusion-awared object opacity
        object_opacity = self.occlusion_opacity(z_vals, transmittance, dists, sdf_raw).sum(-1).transpose(0, 1)


        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
        semantic_values = torch.sum(weights.unsqueeze(-1)*semantic, 1)
        depth_values = torch.sum(weights * z_vals, 1, keepdims=True) / (weights.sum(dim=1, keepdims=True) +1e-8)
        # we should scale rendered distance to depth along z direction
        depth_values = depth_scale * depth_values
        
        # white background assumption
        if self.white_bkgd:
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1. - acc_map[..., None]) * self.bg_color.unsqueeze(0)

        output = {
            'rgb':rgb,
            'semantic_values': semantic_values, # here semantic value calculated as in ObjectSDF
            'object_opacity': object_opacity, 
            'rgb_values': rgb_values,
            'depth_values': depth_values,
            'z_vals': z_vals,
            'depth_vals': z_vals * depth_scale,
            'sdf': sdf.reshape(z_vals.shape),
            'weights': weights,
        }

        if self.training:
            # Sample points for the eikonal loss
            n_eik_points = batch_size * num_pixels 
            
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-self.scene_bounding_sphere, self.scene_bounding_sphere).cuda()

            # add some of the near surface points
            eik_near_points = (cam_loc.unsqueeze(1) + z_samples_eik.unsqueeze(2) * ray_dirs.unsqueeze(1)).reshape(-1, 3)
            eikonal_points = torch.cat([eikonal_points, eik_near_points], 0)
            # add some neighbour points as unisurf
            neighbour_points = eikonal_points + (torch.rand_like(eikonal_points) - 0.5) * 0.01   
            eikonal_points = torch.cat([eikonal_points, neighbour_points], 0)
            
            grad_theta = self.implicit_network.gradient(eikonal_points)

            sample_sdf = self.implicit_network.get_sdf_raw(eikonal_points)
            sdf_value = self.implicit_network.get_sdf_vals(eikonal_points)
            output['sample_sdf'] = sample_sdf
            output['sample_minsdf'] = sdf_value
            
            # split gradient to eikonal points and heighbour ponits
            output['grad_theta'] = grad_theta[:grad_theta.shape[0]//2]
            output['grad_theta_nei'] = grad_theta[grad_theta.shape[0]//2:]
        
        # compute normal map
        normals = gradients / (gradients.norm(2, -1, keepdim=True) + 1e-6)
        normals = normals.reshape(-1, N_samples, 3)
        normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)
        
        # transform to local coordinate system
        rot = pose[0, :3, :3].permute(1, 0).contiguous()
        normal_map = rot @ normal_map.permute(1, 0)
        normal_map = normal_map.permute(1, 0).contiguous()
        
        output['normal_map'] = normal_map

        return output

    def volume_rendering(self, z_vals, sdf):
        density_flat = self.density(sdf)
        density = density_flat.reshape(-1, z_vals.shape[1])  # (batch_size * num_pixels) x N_samples

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)  # shift one step
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        weights = alpha * transmittance # probability of the ray hits something here

        return weights, transmittance, dists

    def occlusion_opacity(self, z_vals, transmittance, dists, sdf_raw):
        obj_density = self.density(sdf_raw).transpose(0, 1).reshape(-1, dists.shape[0], dists.shape[1]) # [#object, #ray, #sample points]       
        free_energy = dists * obj_density
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        object_weight = alpha * transmittance
        return object_weight