import torch
from torch import nn
import utils.general as utils
import math
import torch.nn.functional as F

# copy from MiDaS
def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


def convert_number_to_digits(x):
    assert isinstance(x, int), 'the input value {} should be int'.format(x)
    v = 2**x
    # convert to 0-1 digits

    


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)
# end copy
    

class MonoSDFLoss(nn.Module):
    def __init__(self, rgb_loss, 
                 eikonal_weight, 
                 smooth_weight = 0.005,
                 depth_weight = 0.1,
                 normal_l1_weight = 0.05,
                 normal_cos_weight = 0.05,
                 end_step = -1):
        super().__init__()
        self.eikonal_weight = eikonal_weight
        self.smooth_weight = smooth_weight
        self.depth_weight = depth_weight
        self.normal_l1_weight = normal_l1_weight
        self.normal_cos_weight = normal_cos_weight
        self.rgb_loss = utils.get_class(rgb_loss)(reduction='mean')
        
        self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)
        
        # print(f"using weight for loss RGB_1.0 EK_{self.eikonal_weight} SM_{self.smooth_weight} Depth_{self.depth_weight} NormalL1_{self.normal_l1_weight} NormalCos_{self.normal_cos_weight}")
        
        self.step = 0
        self.end_step = end_step

    def get_rgb_loss(self,rgb_values, rgb_gt):
        rgb_gt = rgb_gt.reshape(-1, 3)
        rgb_loss = self.rgb_loss(rgb_values, rgb_gt)
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_smooth_loss(self,model_outputs):
        # smoothness loss as unisurf
        g1 = model_outputs['grad_theta']
        g2 = model_outputs['grad_theta_nei']
        
        normals_1 = g1 / (g1.norm(2, dim=1).unsqueeze(-1) + 1e-5)
        normals_2 = g2 / (g2.norm(2, dim=1).unsqueeze(-1) + 1e-5)
        smooth_loss =  torch.norm(normals_1 - normals_2, dim=-1).mean()
        return smooth_loss
    
    def get_depth_loss(self, depth_pred, depth_gt, mask):
        # TODO remove hard-coded scaling for depth
        return self.depth_loss(depth_pred.reshape(1, 32, 32), (depth_gt * 50 + 0.5).reshape(1, 32, 32), mask.reshape(1, 32, 32))
        
    def get_normal_loss(self, normal_pred, normal_gt):
        normal_gt = torch.nn.functional.normalize(normal_gt, p=2, dim=-1)
        normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1)
        l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1).mean()
        cos = (1. - torch.sum(normal_pred * normal_gt, dim = -1)).mean()
        return l1, cos
        
    def forward(self, model_outputs, ground_truth):
        # import pdb; pdb.set_trace()
        rgb_gt = ground_truth['rgb'].cuda()
        # monocular depth and normal
        depth_gt = ground_truth['depth'].cuda()
        normal_gt = ground_truth['normal'].cuda()
        
        depth_pred = model_outputs['depth_values']
        normal_pred = model_outputs['normal_map'][None]
        
        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt)
        
        if 'grad_theta' in model_outputs:
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        else:
            eikonal_loss = torch.tensor(0.0).cuda().float()

        # only supervised the foreground normal
        mask = ((model_outputs['sdf'] > 0.).any(dim=-1) & (model_outputs['sdf'] < 0.).any(dim=-1))[None, :, None]
        # combine with GT
        mask = (ground_truth['mask'] > 0.5).cuda() & mask

        depth_loss = self.get_depth_loss(depth_pred, depth_gt, mask) if self.depth_weight > 0 else torch.tensor(0.0).cuda().float()
        if isinstance(depth_loss, float):
            depth_loss = torch.tensor(0.0).cuda().float()    
        
        normal_l1, normal_cos = self.get_normal_loss(normal_pred * mask, normal_gt)
        
        smooth_loss = self.get_smooth_loss(model_outputs)
        
        # compute decay weights 
        if self.end_step > 0:
            decay = math.exp(-self.step / self.end_step * 10.)
        else:
            decay = 1.0
            
        self.step += 1

        loss = rgb_loss + \
               self.eikonal_weight * eikonal_loss +\
               self.smooth_weight * smooth_loss +\
               decay * self.depth_weight * depth_loss +\
               decay * self.normal_l1_weight * normal_l1 +\
               decay * self.normal_cos_weight * normal_cos               
        
        output = {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'smooth_loss': smooth_loss,
            'depth_loss': depth_loss,
            'normal_l1': normal_l1,
            'normal_cos': normal_cos
        }

        return output


class ObjectSDFPlusLoss(MonoSDFLoss):
    def __init__(self, rgb_loss, 
                 eikonal_weight,
                 semantic_weight = 0.04,
                 smooth_weight = 0.005,
                 semantic_loss = torch.nn.CrossEntropyLoss(ignore_index = -1),
                 depth_weight = 0.1,
                 normal_l1_weight = 0.05,
                 normal_cos_weight = 0.05,
                 reg_vio_weight = 0.1,
                 use_obj_opacity = True,
                 bg_reg_weight = 0.1,
                 end_step = -1):
        super().__init__(
                 rgb_loss = rgb_loss, 
                 eikonal_weight = eikonal_weight, 
                 smooth_weight = smooth_weight,
                 depth_weight = depth_weight,
                 normal_l1_weight = normal_l1_weight,
                 normal_cos_weight = normal_cos_weight,
                 end_step = end_step)
        self.semantic_weight = semantic_weight
        self.bg_reg_weight = bg_reg_weight
        self.semantic_loss = utils.get_class(semantic_loss)(reduction='mean') if semantic_loss is not torch.nn.CrossEntropyLoss else torch.nn.CrossEntropyLoss(ignore_index = -1)
        self.reg_vio_weight = reg_vio_weight
        self.use_obj_opacity = use_obj_opacity

        print(f"[INFO]: using weight for loss RGB_1.0 EK_{self.eikonal_weight} SM_{self.smooth_weight} Depth_{self.depth_weight} NormalL1_{self.normal_l1_weight} NormalCos_{self.normal_cos_weight}\
            Semantic_{self.semantic_weight}, semantic_loss_type_{self.semantic_loss} Use_object_opacity_{self.use_obj_opacity}")

    def get_semantic_loss(self, semantic_value, semantic_gt):
        semantic_gt = semantic_gt.squeeze()
        semantic_loss = self.semantic_loss(semantic_value, semantic_gt)
        return semantic_loss

    # violiation loss
    def get_violation_reg_loss(self, sdf_value):
        # turn to vector, sdf_value: [#rays, #objects]
        min_value, min_indice = torch.min(sdf_value, dim=1, keepdims=True)
        input = -sdf_value-min_value.detach() # add the min value for all tensor
        res = torch.relu(input).sum(dim=1, keepdims=True) - torch.relu(torch.gather(input, 1, min_indice))
        loss = res.sum()
        return loss
    
    def object_distinct_loss(self, sdf_value, min_sdf):
        _, min_indice = torch.min(sdf_value.squeeze(), dim=1, keepdims=True)
        input = -sdf_value.squeeze() - min_sdf.detach()
        res = torch.relu(input).sum(dim=1, keepdims=True) - torch.relu(torch.gather(input, 1, min_indice))
        loss = res.mean()
        return loss

    def object_opacity_loss(self, predict_opacity, gt_opacity, weight=None):
        # normalize predict_opacity
        # predict_opacity = torch.nn.functional.normalize(predict_opacity, p=1, dim=-1)
        target = torch.nn.functional.one_hot(gt_opacity.squeeze(), num_classes=predict_opacity.shape[1]).float()
        if weight is None:
            loss = F.binary_cross_entropy(predict_opacity.clamp(1e-4, 1-1e-4), target)
        return loss

    # background regularization loss following the desing in Sec 3.2 of RICO (https://arxiv.org/pdf/2303.08605.pdf)
    def bg_tv_loss(self, depth_pred, normal_pred, gt_mask):
        # the depth_pred and normal_pred should form a patch in image space, depth_pred: [ray, 1], normal_pred: [ray, 3], gt_mask: [1, ray, 1]
        size = int(math.sqrt(gt_mask.shape[1]))
        mask = gt_mask.reshape(size, size, -1) 
        depth = depth_pred.reshape(size, size, -1)
        normal = torch.nn.functional.normalize(normal_pred, p=2, dim=-1).reshape(size, size, -1)
        loss = 0
        for stride in [1, 2, 4]:
            hd_d = torch.abs(depth[:, :-stride, :] - depth[:, stride:, :])
            wd_d = torch.abs(depth[:-stride, :, :] - depth[stride:, :, :])
            hd_n = torch.abs(normal[:, :-stride, :] - normal[:, stride:, :])
            wd_n = torch.abs(normal[:-stride, :, :] - normal[stride:, :, :])
            loss+= torch.mean(hd_d*mask[:, :-stride, :]) + torch.mean(wd_d*mask[:-stride, :, :])
            loss+= torch.mean(hd_n*mask[:, :-stride, :]) + torch.mean(wd_n*mask[:-stride, :, :])
        return loss
    

    def forward(self, model_outputs, ground_truth, call_reg=False, call_bg_reg=False):
        output = super().forward(model_outputs, ground_truth)
        if 'semantic_values' in model_outputs and not self.use_obj_opacity: # ObjectSDF loss: semantic field + cross entropy
            semantic_gt = ground_truth['segs'].cuda().long()
            semantic_loss = self.get_semantic_loss(model_outputs['semantic_values'], semantic_gt)
        elif "object_opacity" in model_outputs and self.use_obj_opacity: # ObjectSDF++ loss: occlusion-awared object opacity + MSE
            semantic_gt = ground_truth['segs'].cuda().long()
            semantic_loss = self.object_opacity_loss(model_outputs['object_opacity'], semantic_gt)
        else:
            semantic_loss = torch.tensor(0.0).cuda().float()
        
        if "sample_sdf" in model_outputs and call_reg:
            sample_sdf_loss = self.object_distinct_loss(model_outputs["sample_sdf"], model_outputs["sample_minsdf"])
        else:
            sample_sdf_loss = torch.tensor(0.0).cuda().float()

        background_reg_loss = torch.tensor(0.0).cuda().float()        
        # if call_bg_reg:
        #     mask = (ground_truth['segs'] != 0).cuda()
        #     background_reg_loss = self.bg_tv_loss(model_outputs['bg_depth_values'], model_outputs['bg_normal_map'], mask)
        # else:
        #     background_reg_loss = torch.tensor(0.0).cuda().float()
        output['semantic_loss'] = semantic_loss
        output['collision_reg_loss'] = sample_sdf_loss
        output['background_reg_loss'] = background_reg_loss
        output['loss'] = output['loss'] + self.semantic_weight * semantic_loss + self.reg_vio_weight* sample_sdf_loss
            # + self.bg_reg_weight * background_reg_loss # <- this one is not used in the paper, but is helpful for background regularization
        return output