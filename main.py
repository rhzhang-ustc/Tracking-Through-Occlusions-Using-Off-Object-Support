import flyingthingsdataset
from flow_detection import flow_detect, linker, trajs_compare
from io_preprocess import build_input_matrix
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

# S=8, N=256
# sample['rgbs']: (1, S, 3, 368, 496), every S means a picture of shape(3, , ,)
# sample['masks']: (1, S, 1, 368, 496), every sample has a mask to indicate where is the object
# sample['trajs']: (1, S, N, 2), every sample frame has N points that have trajectory, (x, y)
# sample['valids']: (1, S, N), 0 or 1.0
# sample['visibles']: (1, S, N), 0 or 1.0
# sample['updated_fails']: (1, ), False or True


# super parameters
B = 1
S = 8
N = 256  # S, N, D = trajs.shape, D=2
crop_size = (368, 496)

force_double_inb = False
force_all_inb = False

consistency_threshold = 1
lifespan=3


# datasets
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


train_dataset = flyingthingsdataset.FlyingThingsDataset(
    dset='TRAIN', subset='A',
    use_augs=False,
    N=N, S=S,
    crop_size=crop_size,
    version='ab',
    force_double_inb=force_double_inb,
    force_all_inb=force_all_inb)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=B,
    shuffle=True,
    num_workers=1,
    worker_init_fn=worker_init_fn,
    drop_last=True)
train_iterloader = iter(train_dataloader)


# while run: make sure batch size = 1
for sample in train_iterloader:

    # find some availabel trace
    # implementation in style of 'particles'
    forward_flow_list = flow_detect(sample) # shape(1792, 3), every row is [pt0, pt1, begin_frame]
    backward_flow_list = flow_detect(sample, False)

    forward_link = linker(forward_flow_list,
                          S=S, N=N,
                          lifespan=lifespan,
                          consistency_threshold=consistency_threshold)
    backward_link = linker(backward_flow_list,
                           S=S, N=N,
                           lifespan=lifespan,
                           consistency_threshold=consistency_threshold)

    long_trajs = trajs_compare(forward_link, backward_link, S=S)    # useful trajs with span=lifespan

    # generate input matrix
    input_matrix = build_input_matrix(sample, long_trajs)


    break


