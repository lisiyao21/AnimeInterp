# dataloader for multi frames (acceleration), modified from superslomo

import torch.utils.data as data
from PIL import Image
import os
import os.path
import random
import sys
import numpy as np
import torch
import cv2
import torch.nn.functional as F

def _make_dataset(dir, dirf):
    framesPath = []
    flowPath = []
    # Find and loop over all the clips in root `dir`.
    for index, folder in enumerate(os.listdir(dir)):
        clipsFolderPath = os.path.join(dir, folder)
        flowFolderPath = os.path.join(dirf, folder)
        # Skip items which are not folders.
        if not (os.path.isdir(clipsFolderPath)):
            continue
        framesPath.append([])
        flowPath.append([])
        # Find and loop over all the frames inside the clip.
        # for image in sorted(os.listdir(clipsFolderPath)):
        #     # Add path to list.

        

        for image in sorted(os.listdir(clipsFolderPath)):
            # Add path to list.
            framesPath[index].append(os.path.join(clipsFolderPath, image))
        #framesPath[index].append(os.path.join(clipsFolderPath, 'frame0.jpg'))
        # framesPath[index].append(os.path.join(clipsFolderPath, 'frame1.jpg'))

        # # framesPath[index].append(os.path.join(clipsFolderPath, 'framet1.jpg'))
        # # framesPath[index].append(os.path.join(clipsFolderPath, 'framet2.jpg'))
        # # framesPath[index].append(os.path.join(clipsFolderPath, 'framet3.jpg'))
        # # framesPath[index].append(os.path.join(clipsFolderPath, 'framet4.jpg'))
        # # framesPath[index].append(os.path.join(clipsFolderPath, 'framet5.jpg'))
        # # framesPath[index].append(os.path.join(clipsFolderPath, 'framet6.jpg'))
        # # framesPath[index].append(os.path.join(clipsFolderPath, 'framet7.jpg'))

        # framesPath[index].append(os.path.join(clipsFolderPath, 'frame2.jpg'))
        # framesPath[index].append(os.path.join(clipsFolderPath, 'frame3.jpg'))

        flowPath[index].append(os.path.join(flowFolderPath, 'guide_flo13.npy'))
        # guide_flo31.npy
        flowPath[index].append(os.path.join(flowFolderPath, 'guide_flo31.npy'))
    # print(framesPath)
    return framesPath, flowPath



def _pil_loader(path, cropArea=None, resizeDim=None, frameFlip=0):
    with open(path, 'rb') as f:
        img = Image.open(f)
        # Resize image if specified.
        resized_img = img.resize(resizeDim, Image.ANTIALIAS) if (resizeDim != None) else img
        # cv2.imwrite(resize)
        # Crop image if crop area specified.
        if cropArea != None:
            cropped_img = resized_img.crop(cropArea)
        else:
            cropped_img = resized_img
        # Flip image horizontally if specified.
        flipped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT) if frameFlip else cropped_img


        return flipped_img.convert('RGB')

def _flow_loader(path, cropArea=None, resizeDim=None, frameFlip=0, shiftX=0, shiftY=0):
    flow_np = np.load(path)
    flow = torch.from_numpy(flow_np) 
    # print(flow.size())
    flow = F.interpolate(flow[None], size=(resizeDim[1],resizeDim[0]))[0]

    if resizeDim is not None:
        factor0 = float(resizeDim[0]) / flow_np.shape[2]
        factor1 = float(resizeDim[1]) / flow_np.shape[1]

        flow[0, :, :] *= factor0
        flow[1, :, :] *= factor1
    
    if cropArea is not None:
        flow = flow[:, cropArea[1]:cropArea[3], cropArea[0]:cropArea[2]]

    flow[0] -= shiftX
    flow[1] -= shiftY

    if frameFlip:
        flow = torch.flip(flow, [2])
        flow[0] *= -1
    return flow.clone().detach()
    
    
class AniTripletWithSGMFlow(data.Dataset):
    def __init__(self, root, root_flow, transform=None, resizeSize=(640, 360), randomCropSize=(352, 352), train=True, shift=0):
        # Populate the list with image paths for all the
        # frame in `root`.
        framesPath, flowPath = _make_dataset(root, root_flow)
        # Raise error if no images found in root.
        if len(framesPath) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))
                
        self.randomCropSize = randomCropSize
        self.cropX0         = resizeSize[0] - randomCropSize[0]
        self.cropY0         = resizeSize[1] - randomCropSize[1]
        self.root           = root
        self.transform      = transform
        self.train          = train
        self.resizeSize     = resizeSize
        self.framesPath     = framesPath
        self.flowPath       = flowPath
        self.shift          = shift

    def __getitem__(self, index):
        sample = []
        flow = []
        inter = None
        cropArea = []
        shifts = []
        
        if (self.train):
            ### Data Augmentation ###
            # To select random 9 frames from 12 frames in a clip
            firstFrame = 0
            # Apply random crop on the 9 input frames

            
            shiftX = random.randint(0, self.shift)//2 * 2
            shiftY = random.randint(0, self.shift)//2 * 2
            shiftX = shiftX * -1 if random.randint(0, 1) > 0 else shiftX
            shiftY = shiftY * -1 if random.randint(0, 1) > 0 else shiftY
            # shiftX = 0
            # shiftY = 0
            # print(shiftX, shiftY)            

            cropX0 = random.randint(max(0, -shiftX), min(self.cropX0 - shiftX, self.cropX0))
            cropY0 = random.randint(max(0, -shiftY), min(self.cropY0, self.cropY0 - shiftY))
            

            cropArea.append((cropX0, cropY0, cropX0 + self.randomCropSize[0], cropY0 + self.randomCropSize[1]))
            cropArea.append((cropX0 + shiftX//2, cropY0 + shiftY//2, cropX0 + shiftX//2 + self.randomCropSize[0], cropY0 + shiftY//2 + self.randomCropSize[1]))
            cropArea.append((cropX0 + shiftX, cropY0 + shiftY, cropX0 + shiftX + self.randomCropSize[0], cropY0 + shiftY + self.randomCropSize[1]))
            
            shifts.append((shiftX, shiftY))
            shifts.append((-shiftX, -shiftY))
            # inter = random.randint(1)
            inter = 1
            reverse = random.randint(0, 1)
            if reverse:
                frameRange = [2, 1, 0]
                flowRange = [1, 0]
                inter = 1
                # returnIndex = IFrameIndex - firstFrame - 1
            else:
                frameRange = [0, 1, 2]
                flowRange = [0, 1]
            randomFrameFlip = random.randint(0, 1)
                # returnIndex = firstFrame - IFrameIndex + 7
            # Random flip frametrain_crop_size, train=True, shift=config.shift)
            # reverse = False
            # Fixed settings to return same samples every epoch.
            # For validation/test sets.
            # firstFrame = 0
        else:
            cropArea.append((0, 0, self.randomCropSize[0], self.randomCropSize[1]))
            cropArea.append((0, 0, self.randomCropSize[0], self.randomCropSize[1]))
            cropArea.append((0, 0, self.randomCropSize[0], self.randomCropSize[1]))
            # IFrameIndex = ((index) % 7  + 1)
            # returnIndex = IFrameIndex - 1
            frameRange = [0, 1, 2]
            flowRange = [0, 1]
            randomFrameFlip = 0
            inter = 1
            shifts = [(0, 0), (0, 0)]
        
        # Loop over for all frames corresponding to the `index`.
        for frameIndex in frameRange:
            # Open image using pil and augment the image.

            image = _pil_loader(self.framesPath[index][frameIndex], cropArea=cropArea[frameIndex],  resizeDim=self.resizeSize, frameFlip=randomFrameFlip)
            # image.save(str(frameIndex) + '.jpg')

            # Apply transformation if specified.
            if self.transform is not None:
                image = self.transform(image)
            sample.append(image)
        for flowIndex in flowRange:
            flowa = _flow_loader(self.flowPath[index][flowIndex], cropArea[flowIndex*2], self.resizeSize, randomFrameFlip, shifts[flowIndex][0], shifts[flowIndex][1])
            flow.append(flowa)
        # print(shiftX, shiftY, randomFrameFlip, reverse)
        # flowBA = _flow_loader(self.flowPath[index][flowRange[1]], cropArea[flowRange[1]], self.resizeSize, randomFrameFlip, -shiftX, -shiftY)

        # flow.append(flowAB)
        # flow.append(flowBA)

        # while True:
        #     pass
        t =  0.5
        # while True:
        #     pass
        if self.train:
            return sample, flow
        else:
            return sample, flow

    def __len__(self):
        return len(self.framesPath)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
