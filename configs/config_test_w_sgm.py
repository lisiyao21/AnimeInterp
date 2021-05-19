
testset_root = 'datasets/test_2k_540p'
test_flow_root = 'datasets/test_2k_pre_calc_sgm_flows'
test_annotation_root = 'datasets/test_2k_annotations'

test_size = (960, 540)
test_crop_size = (960, 540)

mean = [0., 0., 0.]
std  = [1, 1, 1]

inter_frames = 1

model = 'AnimeInterp'
pwc_path = None

checkpoint = 'checkpoints/anime_interp_full.ckpt'

store_path = 'outputs/avi_full_results'



