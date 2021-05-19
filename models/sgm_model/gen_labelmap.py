import os, sys
import argparse

import numpy as np
import cv2
from skimage import filters


from linefiller.thinning import thinning
from linefiller.trappedball_fill import trapped_ball_fill_multi, flood_fill_multi, mark_fill, build_fill_map, merge_fill, \
    show_fill_map, my_merge_fill


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





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input_root')
    parser.add_argument('output_root')
    parser.add_argument('--start_idx', default=0,
                        help='')
    parser.add_argument('--end_idx', default=None,
                        help='')

    parser.add_argument('--height', default=960,
                    help='height of the generated flow, default: 960')
    parser.add_argument('--width', default=540,
                    help='width of the generated flow, default: 540')

    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()

    ######
    folder_root = args.input_root
    save_root = args.output_root
    use_gpu = args.use_gpu

    start_idx = int(args.start_idx)
    end_idx = None if args.end_idx is None else int(args.end_idx)

    # tar_size = (1280, 720)
    tar_size = (args.height, args.width)
    # tar_size = (640, 360)
    ######

    print('use gpu: ', use_gpu)
    sys.stdout.flush()

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    folderList = sorted(os.listdir(folder_root))
    print('folderList length: ', len(folderList))
    for f_idx, folder in enumerate(folderList[start_idx:end_idx]):
        f_idx += start_idx

        input_subfolder = os.path.join(folder_root, folder)
        imgFileNames = sorted(os.listdir(input_subfolder))
        print('-- [%d] %s'%(f_idx, folder))
        print(imgFileNames)

        saveFolder = os.path.join(save_root, folder)
        labelMap1_savePath = os.path.join(saveFolder, 'labelmap_1.npy')
        labelMap2_savePath = os.path.join(saveFolder, 'labelmap_3.npy')
        # if os.path.exists(labelMap1_savePath) and os.path.exists(labelMap2_savePath):
        #     try:
        #         binMap1 = np.load(labelMap1_savePath)
        #         binMap3 = np.load(labelMap2_savePath)
        #     except IOError:
        #         print("labelmap file corrupted")
        #     else:
        #         print("already generated")
        #         continue

        sys.stdout.flush()

        img1 = cv2.imread(os.path.join(input_subfolder, imgFileNames[0]))
        img3 = cv2.imread(os.path.join(input_subfolder, imgFileNames[-1]))

        # segmentation
        img1_rs = cv2.resize(img1, tar_size)
        img3_rs = cv2.resize(img3, tar_size)

        if 'Disney' in folder:
            boundImg1 = dline_of(img1_rs, 1, 20, [30,40,30]).astype(np.uint8)
            boundImg3 = dline_of(img3_rs, 1, 20, [30,40,30]).astype(np.uint8)
        else:
            boundImg1 = dline_of(img1_rs, 2, 20, [10,10,10]).astype(np.uint8)
            boundImg3 = dline_of(img3_rs, 2, 20, [10,10,10]).astype(np.uint8)

        ret, binMap1 = cv2.threshold(boundImg1, 220, 255, cv2.THRESH_BINARY)
        ret, binMap3 = cv2.threshold(boundImg3, 220, 255, cv2.THRESH_BINARY)

        print('- trapped_ball_processed()')
        sys.stdout.flush()
        fillMap1 = trapped_ball_processed(binMap1, img1_rs)
        fillMap3 = trapped_ball_processed(binMap3, img3_rs)

        labelMap1 = squeeze_label_map(fillMap1)
        labelMap3 = squeeze_label_map(fillMap3)

        # save flows
        if not os.path.exists(saveFolder):
            os.mkdir(saveFolder)
        
        np.save(labelMap1_savePath, labelMap1)
        np.save(labelMap2_savePath, labelMap3)
        print('save to %s, %s'%(labelMap1_savePath, labelMap2_savePath))
        sys.stdout.flush()

        labelMap1_img = show_fill_map(labelMap1)
        labelMap3_img = show_fill_map(labelMap3)
        cv2.imwrite(os.path.join(saveFolder, 'labelmap_1.jpg'), labelMap1_img)
        cv2.imwrite(os.path.join(saveFolder, 'labelmap_3.jpg'), labelMap3_img)