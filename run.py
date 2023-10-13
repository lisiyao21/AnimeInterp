#! /usr/bin/env python3

import models
import datas
import argparse
import torch
import torchvision
import torchvision.transforms as TF
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from math import log10
import numpy as np
import datetime
from utils.config import Config
import sys
import cv2
from utils.vis_flow import flow_to_color
import json
import tempfile

test_size = (960, 540, 3)
mean = [0., 0., 0.]
std  = [1, 1, 1]
checkpoint = 'checkpoints/anime_interp_full.ckpt'
model_name = 'AnimeInterp'
pwc_path = None

def run(imagefile1, imagefile2, inter_frames, outputdir):
    inter_frames = int(inter_frames)
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    with tempfile.TemporaryDirectory() as td:
        print(td)
        os.mkdir(os.path.join(td, 'images'))
        os.mkdir(os.path.join(td, 'images', '0'))
        os.mkdir(os.path.join(td, 'sgm'))
        os.mkdir(os.path.join(td, 'results'))
        i1 = cv2.imread(imagefile1)
        w = i1.shape[1]
        h = i1.shape[0]
        wtarget = int(w/h*test_size[1])
        print(wtarget)
        i1s = cv2.resize(i1, (wtarget, test_size[1]))
        startx = (test_size[0]-wtarget) // 2
        i1t = np.zeros((test_size[1], test_size[0], test_size[2]))
        i1t[:, startx:startx+wtarget,:] = i1s
        
        cv2.imwrite(os.path.join(td, 'images', '0', 'frame1.png'), i1t)
        i2 = cv2.imread(imagefile2)
        i2s = cv2.resize(i2, (wtarget, test_size[1]))
        i2t = np.zeros((test_size[1], test_size[0], test_size[2]))
        i2t[:, startx:startx+wtarget,:] = i2s
        cv2.imwrite(os.path.join(td, 'images', '0', 'frame3.png'), i2t)
        cv2.imwrite(os.path.join(td, 'images', '0', 'frame2.png'), i2t)
        # run sgm
        os.system("python3 models/sgm_model/gen_sgm.py " + os.path.join(td, 'images') + ' ' + os.path.join(td, 'sgm'))
        # run the big stuff
        normalize1 = TF.Normalize(mean, [1.0, 1.0, 1.0])
        normalize2 = TF.Normalize([0, 0, 0], std)
        trans = TF.Compose([TF.ToTensor(), normalize1, normalize2, ])
        
        revmean = [-x for x in mean]
        revstd = [1.0 / x for x in std]
        revnormalize1 = TF.Normalize([0.0, 0.0, 0.0], revstd)
        revnormalize2 = TF.Normalize(revmean, [1.0, 1.0, 1.0])
        revNormalize = TF.Compose([revnormalize1, revnormalize2])
        
        revtrans = TF.Compose([revnormalize1, revnormalize2, TF.ToPILImage()])
        
        testset = datas.AniTripletWithSGMFlowTest(os.path.join(td, 'images'), os.path.join(td, 'sgm'), trans, test_size[0:2], test_size[0:2], train=False)
        sampler = torch.utils.data.SequentialSampler(testset)
        validationloader = torch.utils.data.DataLoader(testset, sampler=sampler, batch_size=1, shuffle=False, num_workers=1)
        to_img = TF.ToPILImage()
        
        print(testset)
        sys.stdout.flush()
    
        # prepare model
        model = getattr(models, model_name)(pwc_path).cuda()
        model = nn.DataParallel(model)
        retImg = []
    
        # load weights
        dict1 = torch.load(checkpoint)
        model.load_state_dict(dict1['model_state_dict'], strict=False)
    
        # prepare others
        #store_path = config.store_path
    
        ## values for whole image
        psnr_whole = 0
        psnrs = np.zeros([len(testset), inter_frames])
        ssim_whole = 0
        ssims = np.zeros([len(testset), inter_frames])
        ie_whole = 0
        ies = np.zeros([len(testset), inter_frames])
       
        ## values for ROI
        psnr_roi = 0
        ssim_roi = 0
        
        folders = []
    
        print('Everything prepared. Ready to run...')  
        sys.stdout.flush()
    
        #  start testing...
        with torch.no_grad():
            model.eval()
            ii = 0
            for validationIndex, validationData in enumerate(validationloader, 0):
                sample, flow,  index, folder = validationData
                frame0 = None
                frame1 = sample[0]
                frame3 = None
                frame2 = sample[-1]
                folders.append(folder[0][0])
                # initial SGM flow
                F12i, F21i  = flow
                F12i = F12i.float().cuda() 
                F21i = F21i.float().cuda()
                ITs = [sample[tt] for tt in range(1, 2)]
                I1 = frame1.cuda()
                I2 = frame2.cuda()
                #if not os.path.exists(config.store_path + '/' + folder[0][0]):
                #   os.mkdir(config.store_path + '/' + folder[0][0])
                #revtrans(I1.cpu()[0]).save(store_path + '/' + folder[0][0] + '/'  + index[0][0] + '.jpg')
                #revtrans(I2.cpu()[0]).save(store_path + '/' + folder[-1][0] + '/' +  index[-1][0] + '.jpg')
                for tt in range(inter_frames):
                    x = inter_frames
                    t = 1.0/(x+1) * (tt + 1)
                    outputs = model(I1, I2, F12i, F21i, t)
                    It_warp = outputs[0]
                    res = to_img(revNormalize(It_warp.cpu()[0]).clamp(0.0, 1.0))
                    res = cv2.cvtColor(np.asarray(res), cv2.COLOR_BGR2RGB)
                    rescrop = np.zeros((test_size[1], wtarget, 3))
                    rescrop[:,:,:] = res[:, startx:startx+wtarget,:]
                    ress = cv2.resize(rescrop, (w, h))
                    cv2.imwrite(outputdir + '/' +  'frame-' + str(tt) + '.png', ress)
                    #save_flow_to_img(outputs[1].cpu(), store_path + '/' + folder[1][0] + '/' + index[1][0] + '_F12')
                    #save_flow_to_img(outputs[2].cpu(), store_path + '/' + folder[1][0] + '/' + index[1][0] + '_F21')

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("usage: run.py frame1 frame2 num_inter_frames output_dir")
    else:
        run(*sys.argv[1:])