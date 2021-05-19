##################################################
#  RFR is implemented based on RAFT optical flow #
##################################################

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock, SmallUpdateBlock
from .extractor import BasicEncoder, SmallEncoder
from .corr import CorrBlock
from .utils import bilinear_sampler, coords_grid, upflow8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

def backwarp(img, flow):
    _, _, H, W = img.size()

    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]

    gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))

    gridX = torch.tensor(gridX, requires_grad=False,).cuda()
    gridY = torch.tensor(gridY, requires_grad=False,).cuda()
    x = gridX.unsqueeze(0).expand_as(u).float() + u
    y = gridY.unsqueeze(0).expand_as(v).float() + v
    # range -1 to 1
    x = 2*(x/(W-1) - 0.5)
    y = 2*(y/(H-1) - 0.5)
    # stacking X and Y
    grid = torch.stack((x,y), dim=3)
    # Sample pixels using bilinear interpolation.
    imgOut = torch.nn.functional.grid_sample(img, grid)

    return imgOut
class ErrorAttention(nn.Module):
    """A three-layer network for predicting mask"""
    def __init__(self, input, output):
        super(ErrorAttention, self).__init__()
        self.conv1 = nn.Conv2d(input, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(38, output, 3, padding=1)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()

    def forward(self, x1):
        x = self.prelu1(self.conv1(x1))
        x = self.prelu2(torch.cat([self.conv2(x), x1], dim=1)) 
        x = self.conv3(x)
        return x

class RFR(nn.Module):
    def __init__(self, args):
        super(RFR, self).__init__()
        self.attention2 = ErrorAttention(6, 1)
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4
        args.dropout = 0
        self.args = args

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='none', dropout=args.dropout)        
        # self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
        
        

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        H, W = image1.size()[2:4]
        H8 = H // 8 * 8
        W8 = W // 8 * 8

        if flow_init is not None:
            flow_init_resize = F.interpolate(flow_init, size=(H8//8, W8//8), mode='nearest')

            flow_init_resize[:, :1] = flow_init_resize[:, :1].clone() * (W8 // 8 *1.0) / flow_init.size()[3]
            flow_init_resize[:, 1:] = flow_init_resize[:, 1:].clone() * (H8 // 8*1.0) / flow_init.size()[2]
            
            if not hasattr(self.args, 'not_use_rfr_mask') or ( hasattr(self.args, 'not_use_rfr_mask') and (not self.args.not_use_rfr_mask)):
                im18 = F.interpolate(image1, size=(H8//8, W8//8), mode='bilinear')
                im28 = F.interpolate(image2, size=(H8//8, W8//8), mode='bilinear')
                
                warp21 = backwarp(im28, flow_init_resize)
                error21 = torch.sum(torch.abs(warp21 - im18), dim=1, keepdim=True)
                # print('errormin', error21.min(), error21.max())
                f12init = torch.exp(- self.attention2(torch.cat([im18, error21, flow_init_resize], dim=1)) ** 2) * flow_init_resize
        else:
            flow_init_resize = None
            flow_init = torch.zeros(image1.size()[0], 2, image1.size()[2]//8, image1.size()[3]//8).cuda()
            error21 = torch.zeros(image1.size()[0], 1, image1.size()[2]//8, image1.size()[3]//8).cuda()

            f12_init = flow_init
            # print('None inital flow!')
        
        image1 = F.interpolate(image1, size=(H8, W8), mode='bilinear')
        image2 = F.interpolate(image2, size=(H8, W8), mode='bilinear')

        f12s, f12, f12_init = self.forward_pred(image1, image2, iters, flow_init_resize, upsample, test_mode)
        
 
        if (hasattr(self.args, 'requires_sq_flow') and self.args.requires_sq_flow):
            for ii in range(len(f12s)):
                f12s[ii] = F.interpolate(f12s[ii], size=(H, W), mode='bilinear')
                f12s[ii][:, :1] = f12s[ii][:, :1].clone() / (1.0*W8) * W
                f12s[ii][:, 1:] = f12s[ii][:, 1:].clone() / (1.0*H8) * H
            if self.training:
                return f12s
            else:
                return [f12s[-1]], f12_init
        else:
            f12[:, :1] = f12[:, :1].clone() / (1.0*W8) * W
            f12[:, 1:] = f12[:, 1:].clone() / (1.0*H8) * H

            f12 = F.interpolate(f12, size=(H, W), mode='bilinear')
            # print('wo!!')
            return f12, f12_init, error21, 
            
    def forward_pred(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """


        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.fnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            if itr == 0:
                if flow_init is not None:
                    coords1 = coords1 + flow_init
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)


        return flow_predictions, flow_up, flow_init
