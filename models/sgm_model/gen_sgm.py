import os, sys
import argparse

import numpy as np
import cv2
from skimage import filters

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from linefiller.thinning import thinning
from linefiller.trappedball_fill import trapped_ball_fill_multi, flood_fill_multi, mark_fill, build_fill_map, merge_fill, \
    show_fill_map, my_merge_fill

# for super pixelpooling
from torch_scatter import scatter_mean
from torch_scatter import scatter_add

import softsplat
from forward_warp2 import ForwardWarp
from my_models import create_VGGFeatNet
from vis_flow import flow_to_color



def dline_of(x, low_thr=1, high_thr=20, bf_args=[30,40,30]):
    xm = cv2.medianBlur(x, 5)
#     xga = cv2.GaussianBlur(x,(5, 5),cv2.BORDER_DEFAULT)
    xb = cv2.bilateralFilter(x, bf_args[0], bf_args[1], bf_args[2])
#     xb = cv2.bilateralFilter(xb, 20, 60, 10 )
    xg = cv2.cvtColor(xb, cv2.COLOR_RGB2GRAY)
    xl = cv2.Laplacian(xb, ddepth = cv2.CV_32F, ksize=5)
    xgg = xl

    xgg = xgg.astype(np.float32) * (255. / (xgg.astype(np.float32).max() * 1.0))

    xh = filters.apply_hysteresis_threshold(xgg, low_thr, high_thr)

    xgg[xh == False] = 0
    # xgg[xh == True] = 255
    
    xgg1 = xgg.copy() * 20
    xgg1 = np.max(xgg1, axis=2)
    
    return np.clip(255 - xgg1, 0, 255)

def squeeze_label_map(label_map):
    ret_label_map = label_map.copy()
    
    labels, counts = np.unique(ret_label_map, return_counts=True)
    label_orders = np.argsort(counts)
    
    for ord_id, ord_val in enumerate(label_orders):
        mask = (label_map == labels[ord_val])
        ret_label_map[mask] = ord_id
   
    return ret_label_map

def trapped_ball_processed(binary, in_image=None, do_merge=True):
    fills = []
    result = binary

    fill = trapped_ball_fill_multi(result, 3, method='max')
    fills += fill
    result = mark_fill(result, fill)
    print('result num 3: ', len(fills))
    
    fill = trapped_ball_fill_multi(result, 2, method=None)
    fills += fill
    result = mark_fill(result, fill)
    print('result num 2: ', len(fills))
    
    fill = trapped_ball_fill_multi(result, 1, method=None)
    fills += fill
    result = mark_fill(result, fill)
    print('result num 1: ', len(fills))

    fill = flood_fill_multi(result)
    fills += fill
    print('flood_fill_multi num 1: ', len(fills))

    fillmap = build_fill_map(result, fills)
#     print('fillmap num: ', len(np.unique(fillmap)))

    if do_merge:
        if in_image is None:
            fillmap = merge_fill(fillmap, max_iter=10)
        else:
            fillmap = my_merge_fill(in_image, fillmap)
    fillmap = thinning(fillmap)
    
    return fillmap


def superpixel_count(label_map):
    _, pixelCounts = np.unique(label_map, return_counts=True)
    
    return pixelCounts

def mutual_matching(corrMap, descending = True):
    sortedCorrMap_1, ranks_1 = corrMap.sort(dim=1, descending=descending)
    sortedCorrMap_2, ranks_2 = corrMap.sort(dim=0, descending=descending)

    _, idxRanks_1 = ranks_1.sort(dim=1, descending=False)
    _, idxRanks_2 = ranks_2.sort(dim=0, descending=False)

#     print(idxRanks_1.shape)
#     print(idxRanks_2.shape)

    mutualRanks = idxRanks_1 + idxRanks_2
    rankSum_1to2, matching_1to2 = mutualRanks.min(dim=1)
    rankSum_2to1, matching_2to1 = mutualRanks.min(dim=0)

    return (rankSum_1to2, matching_1to2, sortedCorrMap_1, 
        rankSum_2to1, matching_2to1, sortedCorrMap_2)


def superpixel_pooling(feat_map, label_map, use_gpu=False):
    fC,fH,fW = feat_map.shape
    lH,lW = label_map.shape
    if fH != lH or fW != lW:
        print('feature map and label map do not match')
        return
    
    feat_flat = feat_map.reshape([fC,fH*fW])
    label_flat = torch.tensor(label_map.reshape(fH*fW)).long()
    
#     print('max label: ', torch.max(label_flat).item())
#     print('superpxiel num: ', len(torch.unique(label_flat)))
    if use_gpu:
        feat_flat = feat_flat.cuda()
        label_flat = label_flat.cuda()

    poolMean = scatter_mean(feat_flat, label_flat, dim=1)
    
    return poolMean


def get_bounding_rect(points):
    """Get a bounding rect of points.

    # Arguments
        points: array of points.
    # Returns
        rect coord
    """
    x1, y1, x2, y2 = np.min(points[1]), np.min(points[0]), np.max(points[1]), np.max(points[0])
    return x1, y1, x2, y2

def get_deformable_flow(flowObj, img1, mask_1, box_1, 
                                img2, mask_2, box_2, 
                                warp_func=None, use_gpu=False):
    mask1_patch = mask_1[box_1[1]:box_1[3]+1, box_1[0]:box_1[2] +1]
    mask2_patch = mask_2[box_2[1]:box_2[3]+1, box_2[0]:box_2[2]+1 ]

    gray1 =  cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1_patch = gray1[box_1[1]:box_1[3]+1, box_1[0]:box_1[2]+1]
    img1_patch[np.invert(mask1_patch)] = 0
    if np.mean(img1_patch[mask1_patch]) < 20:
        img1_patch[mask1_patch] = 0.8*img1_patch[mask1_patch] + 0.2*200

    gray2 =  cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2_patch = gray2[box_2[1]:box_2[3]+1, box_2[0]:box_2[2]+1 ]
    img2_patch[np.invert(mask2_patch)] = 0
    if np.mean(img2_patch[mask2_patch]) < 20:
        img2_patch[mask2_patch] = 0.8*img2_patch[mask2_patch] + 0.2*200

    # the black border should be larger than 8 required by DIS
    tarH = max(img1_patch.shape[0], img2_patch.shape[0], 16) + 10
    tarW = max(img1_patch.shape[1], img2_patch.shape[1], 16) + 10

    H_front_pad = int((tarH - img1_patch.shape[0])//2)
    H_back_pad = tarH - H_front_pad -img1_patch.shape[0]
    W_front_pad = int((tarW - img1_patch.shape[1])//2)
    W_back_pad = tarW - W_front_pad - img1_patch.shape[1]
    img1_patch_pad = np.pad(img1_patch, ([H_front_pad, H_back_pad], [W_front_pad, W_back_pad]), mode='constant')
    mask1_patch_pad =  np.pad(mask1_patch.astype(np.uint8), ([H_front_pad, H_back_pad], [W_front_pad, W_back_pad]), mode='constant')
    
    H_front_pad2 = int((tarH - img2_patch.shape[0])//2)
    H_back_pad2 = tarH - H_front_pad2 -img2_patch.shape[0]
    W_front_pad2 = int((tarW - img2_patch.shape[1])//2)
    W_back_pad2 = tarW - W_front_pad2 - img2_patch.shape[1]
    img2_patch_pad = np.pad(img2_patch, ([H_front_pad2, H_back_pad2], [W_front_pad2, W_back_pad2]), mode='constant')
    mask2_patch_pad =  np.pad(mask2_patch.astype(np.uint8), ([H_front_pad2, H_back_pad2], [W_front_pad2, W_back_pad2]), mode='constant')
    
    # compute flow between patches
    patch_flow = flowObj.calc(img1_patch_pad,  img2_patch_pad,  None)
    
    union_rate = 1
    if warp_func is not None:
        patch_flow_tensor = torch.Tensor(patch_flow.transpose([2,0,1])).unsqueeze(0)
        fwarp_mask_tensor = torch.Tensor(mask1_patch_pad).unsqueeze(0).unsqueeze(0)
        if use_gpu:
            # use softsplat forward warp
            fwarp_mask_tensor = warp_func(fwarp_mask_tensor.cuda(), patch_flow_tensor.cuda())
        else:
            fwarp_mask_tensor, norm = warp_func(fwarp_mask_tensor, patch_flow_tensor)
            fwarp_mask_tensor[norm > 0] = fwarp_mask_tensor[norm > 0] / norm[norm > 0]
        
        fwarp_mask = fwarp_mask_tensor[0][0].cpu().numpy()

        kernel = np.ones((5,5), np.uint8)
#         fwarp_mask_close = cv2.morphologyEx(fwarp_mask, cv2.MORPH_CLOSE, kernel)
        fwarp_mask_close = fwarp_mask
        fwarp_mask_close[fwarp_mask_close<0.05] = 0
        
        union_region = np.logical_and(fwarp_mask_close.astype(np.bool), mask2_patch_pad.astype(np.bool))
        union_rate = np.sum(union_region.astype(np.bool))/np.sum(fwarp_mask_close.astype(np.bool))

    ###
    mask1_patch_pad = np.pad(mask1_patch, ([H_front_pad, H_back_pad], [W_front_pad, W_back_pad]), mode='constant')
    mask_tmp = np.repeat(mask1_patch_pad[:,:,np.newaxis], 2, axis=2)
    points_in_patch = np.where(mask1_patch_pad)
    
    return patch_flow, points_in_patch, union_rate



def get_guidance_flow(label_map1, label_map2, img1, img2,
                        rank_sum, matching, sorted_corrMap, 
                        mean_X_A, mean_Y_A, mean_X_B, mean_Y_B, 
                        rank_sum_thr=0, use_gpu=False):
    lH, lW = label_map1.shape
    labelNum = len(np.unique(label_map1))
    pixelCounts = superpixel_count(label_map1)
    pixelCounts_2 = superpixel_count(label_map2)

    guideflow_X = np.zeros([lH, lW])
    guideflow_Y = np.zeros([lH, lW])

    color_patch1 = show_fill_map(label_map1)
    color_patch2 = show_fill_map(label_map2)

    flowObj = cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOpticalFlow_PRESET_MEDIUM)
    # flowObj.setUseMeanNormalization(False)
    # flowObj.setUseSpatialPropagation(False)
    flowObj.setVariationalRefinementIterations(25)
    # flowObj.setPatchSize(8)
    # flowObj.setPatchStride(8)
    flowObj.setFinestScale(0) # max 6
    flowObj.setGradientDescentIterations(50)

    if use_gpu:
        func_fwarp2 = softsplat.ModuleSoftsplat('average')
    else:
        func_fwarp2 = ForwardWarp()

    for l_id_1 in range(labelNum):
        # labelMask = (label_map1 == l_id_1)
        pixelNum = pixelCounts[l_id_1]

        l_id_2 = matching[l_id_1].item()
        curFlowX = mean_X_B[l_id_2] - mean_X_A[l_id_1]
        curFlowY = mean_Y_B[l_id_2] - mean_Y_A[l_id_1]
        flowLen = np.linalg.norm([curFlowX, curFlowY])

        labelMask = (label_map1 == l_id_1)
        labelMask2 = (label_map2 == l_id_2)


        pixelNum_2 = pixelCounts_2[l_id_2]
        isAreaValid = (max(pixelNum/pixelNum_2, pixelNum_2/pixelNum) < 3)

        isValidPatch = (rank_sum[l_id_1] <= rank_sum_thr and flowLen <= 250 and 
                        pixelNum < maxPixNum*0.12 and pixelNum > 50 and
                        isAreaValid)


        if not isValidPatch:
            guideflow_X[labelMask] = 0
            guideflow_Y[labelMask] = 0
            for cc in range(3):
                color_patch1[:, :, cc][labelMask] = 255
        else:
            points_1 = np.where(labelMask)
            points_2 = np.where(labelMask2)
            box_1 = get_bounding_rect(points_1)
            box_2 = get_bounding_rect(points_2)


            patch_flow, points_in_patch, union_rate = get_deformable_flow(flowObj, 
                                                        img1, labelMask, box_1, 
                                                        img2, labelMask2, box_2,
                                                        warp_func=func_fwarp2, 
                                                        use_gpu=use_gpu)
            
            if union_rate > 0.8:
                patch_flow_X = patch_flow[:,:,0]
                patch_flow_Y = patch_flow[:,:,1]

                guideflow_X[points_1] = (box_2[0] + box_2[2] - box_1[0] - box_1[2])/2 + patch_flow_X[points_in_patch]
                guideflow_Y[points_1] = (box_2[1] + box_2[3] - box_1[1] - box_1[3])/2 + patch_flow_Y[points_in_patch]

                for cc in range(3):
                    color_patch1[:, :, cc][labelMask] = color_patch2[:, :, cc][labelMask2][0]
            else:
                guideflow_X[labelMask] = 0
                guideflow_Y[labelMask] = 0
                for cc in range(3):
                    color_patch1[:, :, cc][labelMask] = 255

    guideflow = np.concatenate((guideflow_X[np.newaxis,:,:], guideflow_Y[np.newaxis,:,:]), axis=0)
    matching_color_patch = np.hstack((color_patch1, color_patch2)).astype(np.uint8)

    return guideflow, matching_color_patch


def get_ctx_feature(label_map, featx1, featx2, featx4, featx8):
    labelNum = len(np.unique(label_map))
    
    featx1_pad = F.pad(featx1, [64, 64, 64, 64])
    featx2_pad = F.pad(featx2, [32, 32, 32, 32])
    featx4_pad = F.pad(featx4, [16, 16, 16, 16])
    # featx8_pad = F.pad(featx8, [8, 8, 8, 8])
    
    for l_idx in range(labelNum):
        mask = (label_map == l_idx)
        points = np.where(mask)
        box = get_bounding_rect(points)

        # same recepetive field
        box_h = box[3] - box[1] + 64
        box_w = box[2] - box[0] + 64

        featx1_patch = featx1_pad[:,:,box[1]:box[1]+box_h+1, box[0]:box[0]+box_w+1]
        featx2_patch = featx2_pad[:,:,box[1]//2:(box[1]+box_h)//2+1, box[0]//2:(box[0]+box_w)//2+1]
        featx4_patch = featx4_pad[:,:,box[1]//4:(box[1]+box_h)//4+1, box[0]//4:(box[0]+box_w)//4+1]
        # featx8_patch = featx8_pad[:,:,box[1]//8:(box[1]+box_h)//8+1, box[0]//8:(box[0]+box_w)//8+1]

        # average whole patch
        featx1_patch_flat = featx1_patch.flatten(start_dim=2, end_dim=-1).mean(dim=-1)
        featx2_patch_flat = featx2_patch.flatten(start_dim=2, end_dim=-1).mean(dim=-1)
        featx4_patch_flat = featx4_patch.flatten(start_dim=2, end_dim=-1).mean(dim=-1)
        # featx8_patch7x7 = featx8_patch.flatten(start_dim=2, end_dim=-1).mean(dim=-1)
        
        feat_patch_flat = torch.cat([featx1_patch_flat, featx2_patch_flat, featx4_patch_flat], dim=1)

        
        if l_idx == 0:
            ctxFeat = feat_patch_flat
        else:
            ctxFeat = torch.cat([ctxFeat, feat_patch_flat],dim=0)
            
    return ctxFeat






if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input_root')
    parser.add_argument('output_root')

    parser.add_argument('--label_root', default=None, help='root for label maps')
    parser.add_argument('--start_idx', default=0,
                        help='threshold to differ motion regions from static')
    parser.add_argument('--end_idx', default=None,
                        help='threshold to differ motion regions from static')

    parser.add_argument('--rank_sum_thr', default=0,
                        help='threshold for rank sum')
    parser.add_argument('--height', default=960,
                    help='height of the generated flow, default: 960')
    parser.add_argument('--width', default=540,
                    help='width of the generated flow, default: 540')

    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()

    ######
    folder_root = args.input_root
    save_root = args.output_root

    label_root = args.label_root 

    start_idx = int(args.start_idx)
    end_idx = None if args.end_idx is None else int(args.end_idx)
    use_gpu = args.use_gpu

    # tar_size = (1280, 720)
    tar_size = (args.height, args.width)
    # tar_size = (640, 360)

    rankSumThr = int(args.rank_sum_thr)
    ######

    print('use label maps from %s'%label_root)
    print('use gpu: ', use_gpu)
    sys.stdout.flush()

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    ## make models 
    vggNet = create_VGGFeatNet()
    if use_gpu:
        vggNet = vggNet.cuda()

    toTensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])


    totalMatchCount = 0
    folderList = sorted(os.listdir(folder_root))
    if end_idx is None:
        end_idx = len(folderList)

    for f_idx, folder in enumerate(folderList[start_idx:end_idx]):
        f_idx += start_idx
        # if f_idx > 1 + start_idx:
        #     break

        input_subfolder = os.path.join(folder_root, folder)
        imgFileNames = sorted(os.listdir(input_subfolder))
        
        print('-- [%d/%d] %s'%(f_idx, end_idx-1, folder))
        print(imgFileNames)
        sys.stdout.flush()

        img1 = cv2.imread(os.path.join(input_subfolder, imgFileNames[0]))
        img3 = cv2.imread(os.path.join(input_subfolder, imgFileNames[-1]))

        # segmentation
        img1_rs = cv2.resize(img1, tar_size)
        img3_rs = cv2.resize(img3, tar_size)

        if label_root is None:
            if 'Japan' in folder:
                boundImg1 = dline_of(img1_rs, 2, 20, [10,10,10]).astype(np.uint8)
                boundImg3 = dline_of(img3_rs, 2, 20, [10,10,10]).astype(np.uint8)
            else:
                boundImg1 = dline_of(img1_rs, 1, 20, [30,40,30]).astype(np.uint8)
                boundImg3 = dline_of(img3_rs, 1, 20, [30,40,30]).astype(np.uint8)

            ret, binMap1 = cv2.threshold(boundImg1, 220, 255, cv2.THRESH_BINARY)
            ret, binMap3 = cv2.threshold(boundImg3, 220, 255, cv2.THRESH_BINARY)

            print('- trapped_ball_processed()')
            sys.stdout.flush()
            fillMap1 = trapped_ball_processed(binMap1, img1_rs)
            fillMap3 = trapped_ball_processed(binMap3, img3_rs)

            labelMap1 = squeeze_label_map(fillMap1)
            labelMap3 = squeeze_label_map(fillMap3)
        else:
            print('- load labelmap')
            sys.stdout.flush()

            print(os.path.join(label_root, folder, 'labelmap_1.npy'))
            print(os.path.join(label_root, folder, 'labelmap_3.npy'))

            labelMap1 = np.load(os.path.join(label_root, folder, 'labelmap_1.npy'))
            print(labelMap1.shape)
            labelMap3 = np.load(os.path.join(label_root, folder, 'labelmap_3.npy'))
            print(labelMap3.shape)


        # VGG features
        img1_rgb = cv2.cvtColor(img1_rs, cv2.COLOR_BGR2RGB)
        img3_rgb = cv2.cvtColor(img3_rs, cv2.COLOR_BGR2RGB)

        img1_tensor = normalize(toTensor(img1_rgb/255.).float())
        img1_tensor = img1_tensor.unsqueeze(dim=0)
        img3_tensor = normalize(toTensor(img3_rgb/255.).float())
        img3_tensor = img3_tensor.unsqueeze(dim=0)

        if use_gpu:
            img1_tensor = img1_tensor.cuda()
            img3_tensor = img3_tensor.cuda()

        # featx1_1 = vggNet.slice1(img1_tensor)
        # featx1_3 = vggNet.slice1(img3_tensor)
        featx1_1, featx2_1, featx4_1, featx8_1, featx16_1 = vggNet(img1_tensor)
        featx1_3, featx2_3, featx4_3, featx8_3, featx16_3 = vggNet(img3_tensor)

        print('- compute correlation map')
        sys.stdout.flush()

        # superpixel pooling
        labelMap1_x2 = labelMap1[1::2,1::2]
        labelMap1_x4 = labelMap1_x2[1::2,1::2]
        labelMap1_x8 = labelMap1_x4[1::2,1::2]
        # labelMap1_x16 = labelMap1_x8[1::2,1::2]
        labelMap3_x2 = labelMap3[1::2,1::2]
        labelMap3_x4 = labelMap3_x2[1::2,1::2]
        labelMap3_x8 = labelMap3_x4[1::2,1::2]
        # labelMap3_x16 = labelMap3_x8[1::2,1::2]

        featx1_pool_1 = superpixel_pooling(featx1_1[0], labelMap1, use_gpu)
        featx2_pool_1 = superpixel_pooling(featx2_1[0], labelMap1_x2, use_gpu)
        featx4_pool_1 = superpixel_pooling(featx4_1[0], labelMap1_x4, use_gpu)
        featx8_pool_1 = superpixel_pooling(featx8_1[0], labelMap1_x8, use_gpu)
        # featx16_pool_1 = superpixel_pooling(featx16_1[0], labelMap1_x16, use_gpu)
        featx1_pool_3 = superpixel_pooling(featx1_3[0], labelMap3, use_gpu)
        featx2_pool_3 = superpixel_pooling(featx2_3[0], labelMap3_x2, use_gpu)
        featx4_pool_3 = superpixel_pooling(featx4_3[0], labelMap3_x4, use_gpu)
        featx8_pool_3 = superpixel_pooling(featx8_3[0], labelMap3_x8, use_gpu)
        # featx16_pool_3 = superpixel_pooling(featx16_3[0], labelMap3_x16, use_gpu)
        
        feat_pool_1 = torch.cat([featx1_pool_1, featx2_pool_1, featx4_pool_1, featx8_pool_1], dim=0)
        feat_pool_3 = torch.cat([featx1_pool_3, featx2_pool_3, featx4_pool_3, featx8_pool_3], dim=0)

        # normalization
        feat_p1_tmp = feat_pool_1 - feat_pool_1.min(dim=0)[0]
        feat_p1_norm = feat_p1_tmp/feat_p1_tmp.sum(dim=0)
        feat_p3_tmp = feat_pool_3 - feat_pool_3.min(dim=0)[0]
        feat_p3_norm = feat_p3_tmp/feat_p3_tmp.sum(dim=0)


        # for pixel distance
        lH, lW = labelMap1.shape
        gridX, gridY = np.meshgrid(np.arange(lW), np.arange(lH))

        gridX_flat = torch.tensor(gridX.astype(np.float), requires_grad=False).reshape(lH*lW)
        gridY_flat = torch.tensor(gridY.astype(np.float), requires_grad=False).reshape(lH*lW)

        labelMap1_flat = torch.tensor(labelMap1.reshape(lH*lW)).long()
        labelMap3_flat = torch.tensor(labelMap3.reshape(lH*lW)).long()

        if use_gpu:
            gridX_flat = gridX_flat.cuda()
            gridY_flat = gridY_flat.cuda()
            labelMap1_flat = labelMap1_flat.cuda()
            labelMap3_flat = labelMap3_flat.cuda()

        mean_X_1 = scatter_mean(gridX_flat, labelMap1_flat).cpu().numpy()
        mean_Y_1 = scatter_mean(gridY_flat, labelMap1_flat).cpu().numpy()
        mean_X_3 = scatter_mean(gridX_flat, labelMap3_flat).cpu().numpy()
        mean_Y_3 = scatter_mean(gridY_flat, labelMap3_flat).cpu().numpy()

        # pixel count in superpixel
        pixelCounts_1 = superpixel_count(labelMap1)
        pixelCounts_3 = superpixel_count(labelMap3)

        # some other distance
        labelNum_1 = len(np.unique(labelMap1))
        labelNum_3 = len(np.unique(labelMap3))
        print('label num: %d, %d'%(labelNum_1, labelNum_3))

        maxDist = np.linalg.norm([lH,lW])
        maxPixNum = lH*lW

        corrMap = torch.zeros(labelNum_1, labelNum_3)
        ctxSimMap = torch.zeros(labelNum_1, labelNum_3)

        for x in range(labelNum_1):
            for y in range(labelNum_3):
                corrMap[x,y] = torch.sum(torch.min(feat_p1_norm[:,x], feat_p3_norm[:,y]))

                # pixel number as similarity
                num_1 = float(pixelCounts_1[x])
                num_3 = float(pixelCounts_3[y])
                
                sizeDiff = max(num_1/num_3, num_3/num_1)
                if sizeDiff > 3:
                    corrMap[x,y] -= sizeDiff/20

                # spatial distance as similarity
                dist = np.linalg.norm([mean_X_1[x] - mean_X_3[y], mean_Y_1[x] - mean_Y_3[y]])/maxDist
                
                if dist > 0.14:
                    corrMap[x,y] -= dist/5


        matchingMetaData = mutual_matching(corrMap)
        rankSum_1to3, matching_1to3, sortedCorrMap_1 = matchingMetaData[:3]
        rankSum_3to1, matching_3to1, sortedCorrMap_3 = matchingMetaData[3:]

        mMatchCount_1 = (rankSum_1to3 <= rankSumThr).sum()
        mMatchCount_3 = (rankSum_3to1 <= rankSumThr).sum()
        totalMatchCount += (mMatchCount_1 + mMatchCount_3)/2

        print('match count: %d, %d'%(mMatchCount_1, mMatchCount_3))


        print('- generating flows')
        sys.stdout.flush()
        # create flows
        guideflow_1to3, matching_color_1to3 = get_guidance_flow(labelMap1, labelMap3, img1_rs, img3_rs,
                                rankSum_1to3, matching_1to3, sortedCorrMap_1,
                                mean_X_1, mean_Y_1, mean_X_3, mean_Y_3, 
                                rank_sum_thr = rankSumThr, use_gpu = use_gpu)
        guideflow_3to1, matching_color_3to1 = get_guidance_flow(labelMap3, labelMap1, img3_rs, img1_rs,
                                rankSum_3to1, matching_3to1, sortedCorrMap_3.transpose(1,0), 
                                mean_X_3, mean_Y_3, mean_X_1, mean_Y_1, 
                                rank_sum_thr = rankSumThr, use_gpu = use_gpu)

        # save flows
        saveFolder = os.path.join(save_root, folder)
        if not os.path.exists(saveFolder):
            os.mkdir(saveFolder)
        
        flow13_savePath = os.path.join(saveFolder, 'guide_flo13.npy')
        flow31_savePath = os.path.join(saveFolder, 'guide_flo31.npy')
        np.save(flow13_savePath, guideflow_1to3)
        np.save(flow31_savePath, guideflow_3to1)
        print('save to %s, %s'%(flow13_savePath, flow31_savePath))
        sys.stdout.flush()

        flow13_rgb = flow_to_color(guideflow_1to3.transpose([1,2,0]))
        flow31_rgb = flow_to_color(guideflow_3to1.transpose([1,2,0]))
        cv2.imwrite(os.path.join(saveFolder, 'guide_flo13.jpg'), flow13_rgb[:,:,::-1])
        cv2.imwrite(os.path.join(saveFolder, 'guide_flo31.jpg'), flow31_rgb[:,:,::-1])

        cv2.imwrite(os.path.join(saveFolder, 'matching_color_1to3.jpg'), matching_color_1to3)
        cv2.imwrite(os.path.join(saveFolder, 'matching_color_3to1.jpg'), matching_color_3to1)


    print('--- total match counts: %d'%totalMatchCount)

