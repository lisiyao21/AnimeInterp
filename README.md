# AnimeInterp

This is the code for paper "Deep Animation Video Interpolation in the Wild" (CVPR21). 

[Project page] | [[Paper]](https://arxiv.org/abs/2104.02495) | [Data]

> Abstract: In the animation industry, cartoon videos are usually produced at low frame rate since hand drawing of such frames is costly and time-consuming. Therefore, it is desirable to develop computational models that can automatically interpolate the in-between animation frames. However, existing video interpolation methods fail to produce satisfying results on animation data. Compared to natural videos, animation videos possess two unique characteristics that make frame interpolation difficult: 1) cartoons comprise lines and smooth color pieces. The smooth areas lack textures and make it difficult to estimate accurate motions on animation videos. 2) cartoons express stories via exaggeration. Some of the motions are non-linear and extremely large. In this work, we formally define and study the animation video interpolation problem for the first time. To address the aforementioned challenges, we propose an effective framework, AnimeInterp, with two dedicated modules in a coarse-to-fine manner. Specifically, 1) Segment-Guided Matching resolves the "lack of textures" challenge by exploiting global matching among color pieces that are piece-wise coherent. 2) Recurrent Flow Refinement resolves the "non-linear and extremely large motion" challenge by recurrent predictions using a transformer-like architecture. To facilitate comprehensive training and evaluations, we build a large-scale animation triplet dataset, ATD-12K, which comprises 12,000 triplets with rich annotations. Extensive experiments demonstrate that our approach outperforms existing state-of-the-art interpolation methods for animation videos. Notably, AnimeInterp shows favorable perceptual quality and robustness for animation scenarios in the wild. 

![image](https://github.com/lisiyao21/AnimeInterp/blob/main/figs/sample0.png)

## Overview

AnimeInterp consists three parts ('segment-guided mathing', 'recurent flow refine' and 'warping & synthesis') to generate the in-between anime frame given two inputs.

![image](https://github.com/lisiyao21/AnimeInterp/blob/main/figs/pipeline.png)


### Code is coming soon.





