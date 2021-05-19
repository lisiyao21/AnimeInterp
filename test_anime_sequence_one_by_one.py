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
from skimage.measure import compare_psnr, compare_ssim


def save_flow_to_img(flow, des):
        f = flow[0].data.cpu().numpy().transpose([1, 2, 0])
        fcopy = f.copy()
        fcopy[:, :, 0] = f[:, :, 1]
        fcopy[:, :, 1] = f[:, :, 0]
        cf = flow_to_color(-fcopy)
        cv2.imwrite(des + '.jpg', cf)


def validate(config):   
    # preparing datasets & normalization
    normalize1 = TF.Normalize(config.mean, [1.0, 1.0, 1.0])
    normalize2 = TF.Normalize([0, 0, 0], config.std)
    trans = TF.Compose([TF.ToTensor(), normalize1, normalize2, ])

    revmean = [-x for x in config.mean]
    revstd = [1.0 / x for x in config.std]
    revnormalize1 = TF.Normalize([0.0, 0.0, 0.0], revstd)
    revnormalize2 = TF.Normalize(revmean, [1.0, 1.0, 1.0])
    revNormalize = TF.Compose([revnormalize1, revnormalize2])

    revtrans = TF.Compose([revnormalize1, revnormalize2, TF.ToPILImage()])

    testset = datas.AniTripletWithSGMFlowTest(config.testset_root, config.test_flow_root, trans, config.test_size, config.test_crop_size, train=False)
    sampler = torch.utils.data.SequentialSampler(testset)
    validationloader = torch.utils.data.DataLoader(testset, sampler=sampler, batch_size=1, shuffle=False, num_workers=1)
    to_img = TF.ToPILImage()
 
    print(testset)
    sys.stdout.flush()

    # prepare model
    model = getattr(models, config.model)(config.pwc_path).cuda()
    model = nn.DataParallel(model)
    retImg = []

    # load weights
    dict1 = torch.load(config.checkpoint)
    model.load_state_dict(dict1['model_state_dict'], strict=False)

    # prepare others
    store_path = config.store_path

    ## values for whole image
    psnr_whole = 0
    psnrs = np.zeros([len(testset), config.inter_frames])
    ssim_whole = 0
    ssims = np.zeros([len(testset), config.inter_frames])
    ie_whole = 0
    ies = np.zeros([len(testset), config.inter_frames])

    ## values for ROI
    psnr_roi = 0
    ssim_roi = 0
    
    ## values for different levels
    psnrs_level = {'easy':0, 'mid': 0, 'hard':0}
    ssims_level = {'easy':0, 'mid': 0, 'hard':0}
    num_level = {'easy':0, 'mid': 0, 'hard':0}

    ## difficulty level dict
    diff = {0:'easy', 1:'mid', 2:'hard'}    

    folders = []

    print('Everything prepared. Ready to test...')  
    sys.stdout.flush()

    #  start testing...
    with torch.no_grad():
        model.eval()
        ii = 0
        for validationIndex, validationData in enumerate(validationloader, 0):
            print('Testing {}/{}-th group...'.format(validationIndex, len(testset)))
            sys.stdout.flush()
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
            
            if not os.path.exists(config.store_path + '/' + folder[0][0]):
                os.mkdir(config.store_path + '/' + folder[0][0])


            revtrans(I1.cpu()[0]).save(store_path + '/' + folder[0][0] + '/'  + index[0][0] + '.jpg')
            revtrans(I2.cpu()[0]).save(store_path + '/' + folder[-1][0] + '/' +  index[-1][0] + '.jpg')
            for tt in range(config.inter_frames):
                x = config.inter_frames
                t = 1.0/(x+1) * (tt + 1)
                
                outputs = model(I1, I2, F12i, F21i, t)

                It_warp = outputs[0]

                to_img(revNormalize(It_warp.cpu()[0]).clamp(0.0, 1.0)).save(store_path + '/' + folder[1][0] + '/' + index[1][0] + '.png')
                
                save_flow_to_img(outputs[1].cpu(), store_path + '/' + folder[1][0] + '/' + index[1][0] + '_F12')
                save_flow_to_img(outputs[2].cpu(), store_path + '/' + folder[1][0] + '/' + index[1][0] + '_F21')

                estimated = revNormalize(It_warp[0].cpu()).clamp(0.0, 1.0).numpy().transpose(1, 2, 0)
                gt = revNormalize(ITs[tt][0]).clamp(0.0, 1.0).numpy().transpose(1, 2, 0) 
                
                labelFilePath = os.path.join(config.test_annotation_root,
                                            folder[1][0], '%s.json'%folder[1][0])
                
                # crop region of interest
                with open(labelFilePath, 'r') as f:
                    jsonObj = json.load(f)
                    motion_RoI = jsonObj["motion_RoI"]
                    level = jsonObj["level"]

                tempSize = jsonObj["image_size"]
                scaleH = float(tempSize[1])/config.test_size[1]
                scaleW = float(tempSize[0])/config.test_size[0]

                RoI_x = int(jsonObj["motion_RoI"]['x'] // scaleW)
                RoI_y = int(jsonObj["motion_RoI"]['y'] // scaleH)
                RoI_W = int(jsonObj["motion_RoI"]['width'] // scaleW)
                RoI_H = int(jsonObj["motion_RoI"]['height'] // scaleH)

                print('RoI: %f, %f, %f, %f'%(RoI_x,RoI_y,RoI_W,RoI_H))

                estimated_roi = estimated[RoI_y:RoI_y+RoI_H, RoI_x:RoI_x+RoI_W, :]
                gt_roi = gt[RoI_y:RoI_y+RoI_H, RoI_x:RoI_x+RoI_W, :]

                # whole image value
                this_psnr = compare_psnr(estimated, gt)
                this_ssim = compare_ssim(estimated, gt, multichannel=True, gaussian=True)
                this_ie = np.mean(np.sqrt(np.sum((estimated*255 - gt*255)**2, axis=2)))

                psnrs[validationIndex][tt] = this_psnr
                ssims[validationIndex][tt] = this_ssim
                ies[validationIndex][tt] = this_ie
                
                psnr_whole += this_psnr
                ssim_whole += this_ssim
                ie_whole += this_ie
                outputs = None

                # value for difficulty levels
                psnrs_level[diff[level]] += this_psnr
                ssims_level[diff[level]] += this_ssim
                num_level[diff[level]] += 1

                # roi image value
                this_roi_psnr = compare_psnr(estimated_roi, gt_roi)
                this_roi_ssim = compare_ssim(estimated_roi, gt_roi, multichannel=True, gaussian=True)
                
                psnr_roi += this_roi_psnr
                ssim_roi += this_roi_ssim

        psnr_whole /= (len(testset) * config.inter_frames)
        ssim_whole /= (len(testset) * config.inter_frames)
        ie_whole /= (len(testset) * config.inter_frames)
        psnr_roi /= (len(testset) * config.inter_frames)
        ssim_roi /= (len(testset) * config.inter_frames)
        
        for key in num_level:
            psnrs_level[key] /= (num_level[key] * config.inter_frames)
            ssims_level[key] /= (num_level[key] * config.inter_frames)

    return psnrs, ssims, ies, psnr_whole, ssim_whole, psnr_roi, ssim_roi, psnrs_level, ssims_level, folders



if __name__ == "__main__":

    # loading configures
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    config = Config.from_file(args.config)


    if not os.path.exists(config.store_path):
        os.mkdir(config.store_path)



    psnrs, ssims, ies, psnr, ssim, psnr_roi, ssim_roi, psnrs_level, ssims_level, folder = validate(config)
    for ii in range(config.inter_frames):
        print('PSNR of validation frame' + str(ii+1) + ' is {}'.format(np.mean(psnrs[:, ii])))

    for ii in range(config.inter_frames):
        print('PSNR of validation frame' + str(ii+1) + ' is {}'.format(np.mean(ssims[:, ii])))
            
    for ii in range(config.inter_frames):
        print('PSNR of validation frame' + str(ii+1) + ' is {}'.format(np.mean(ies[:, ii])))
            
    print('Whole PSNR is {}'.format(psnr) )
    print('Whole SSIM is {}'.format(ssim) )

    print('ROI PSNR is {}'.format(psnr_roi) )
    print('ROI SSIM is {}'.format(ssim_roi) )

    print('PSNRs for difficulties are {}'.format(psnrs_level) )
    print('SSIMs for difficulties are {}'.format(ssims_level) )

    with open(config.store_path + '/psnr.txt', 'w') as f:
        for index in sorted(range(len(psnrs[:, 0])), key=lambda k: psnrs[k, 0]):
            f.write("{}\t{}\n".format(folder[index], psnrs[index, 0]))

    with open(config.store_path + '/ssim.txt', 'w') as f:
        for index in sorted(range(len(ssims[:, 0])), key=lambda k: ssims[k, 0]):
            f.write("{}\t{}\n".format(folder[index], ssims[index, 0]))

    with open(config.store_path + '/ie.txt', 'w') as f:
        for index in sorted(range(len(ies[:, 0])), key=lambda k: ies[k, 0]):
            f.write("{}\t{}\n".format(folder[index], ies[index, 0]))