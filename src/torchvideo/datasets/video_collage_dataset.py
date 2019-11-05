import os

import numpy as np
from PIL import Image

from .video_dataset import VideoDataset
from .helpers import invoke_transform


class VideoCollageDataset(VideoDataset):

    extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.JPEG']

    def __init__(
            self, root, metafile, sampler=None,
            num_frames=8, frame_interval=None, frame_start=None,
            file_tmpl='{:06d}.jpg', transform=None):
        self.root = root
        self.metafile = metafile
        self.file_tmpl = file_tmpl
        self.sampler = sampler
        self.transform = transform
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.frame_start = frame_start
        self.video_list = self._make_dataset()
        print(f'Number of videos: {len(self.video_list)}')

    def _make_dataset(self):
        videos = []
        with open(self.metafile) as f:
            for line in f.readlines():
                filename, target = line.strip().split(' ')
                path = os.path.join(self.root, filename)
                if any(path.endswith(ext) for ext in self.extensions):
                    videos.append((path, target))
        return videos

    def _load_collage(self, filename, num_frames=None, nrow=8):
        def _is_blank(image):
            return not image.getbbox()

        # TODO: Properly handle non-square image tiles.
        collage = Image.open(filename).convert('RGB')
        collage_w, collage_h = collage.size
        im_w = int(np.floor(collage_w / nrow))
        im_h = im_w

        images = []
        num_images = 0
        for pos_y in range(0, collage_h, im_h):
            for pos_x in range(0, collage_w, im_w):
                area = (pos_x, pos_y, pos_x + im_w, pos_y + im_h)
                image = collage.crop(area)
                images.append(image)
                num_images += 1
        while _is_blank(images[-1]):
            images.pop(-1)
        return images

    def __getitem__(self, index):
        path, target = self.video_list[index]
        all_frames = self._load_collage(path)
        frames_idx = self.sampler.sample(len(all_frames))
        if isinstance(frames_idx, list):
            frames = [all_frames[idx] for idx in frames_idx]
        elif isinstance(frames_idx, slice):
            frames = all_frames[frames_idx]
        else:
            raise ValueError(f'Unrecognized frame_idx type: {type(frames_idx)}')
        frames, target = invoke_transform(self.transform, frames, target)
        return frames, int(target)

    def __len__(self):
        return len(self.video_list)
