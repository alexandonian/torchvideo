import time
import torch
import torchvision

import pretorched
import torchvideo
from torchvideo.datasets import VideoCollageDataset
from torchvideo.samplers import ClipSampler

num_frames = 32
input_mean = [0.5, 0.5, 0.5]
input_std = [0.5, 0.5, 0.5]
ROOT = '/data/vision/oliva/scratch/datasets/kinetics/collages/'
META = '/data/vision/oliva/scratch/datasets/kinetics/collage_list.txt'
transform = torchvision.transforms.Compose([
    torchvideo.transforms.RandomResizedCropVideo(224),
    torchvideo.transforms.CollectFrames(),
    torchvideo.transforms.PILVideoToTensor(),
    torchvideo.transforms.NormalizeVideo(input_mean, input_std),
])
sampler = torchvideo.samplers.ClipSampler(num_frames)
dataset = VideoCollageDataset(ROOT, META, sampler=sampler, transform=transform)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=96,
    shuffle=True,
    num_workers=50,
    pin_memory=False)

data_time = pretorched.runners.utils.AverageMeter('Time')

end = time.time()
for i, (frames, target) in enumerate(dataloader):
    data_time.update(time.time() - end)
    print(i, frames.shape)
    end = time.time()
    print(f'Average loading time: {data_time.avg:.3f}')


