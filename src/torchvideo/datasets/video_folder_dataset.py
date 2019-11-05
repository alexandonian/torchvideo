import os
import torch
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

from PIL.Image import Image

from torchvideo.internal.readers import _get_videofile_frame_count, _is_video_file
from torchvideo.samplers import FrameSampler, _default_sampler
from torchvideo.transforms import PILVideoToTensor

from .helpers import invoke_transform
from .label_sets import LabelSet
from .types import Label, PILVideoTransform, empty_label
from .video_dataset import VideoDataset


class VideoRecordDataset(VideoDataset):
    """Dataset captured as a list of paths pointing to video files stored on a filesystem.

    The folder hierarchy could look something like this: ::

        root/category1/video1.mp4
        root/category1/video2.mp4
        root/category2/video3.mp4
        root/category2/video4.mp4

        ...
    """

    def __init__(
        self,
        root_path: Union[str, Path],
        record_set: list,
        filter: Optional[Callable[[Path], bool]] = None,
        sampler: FrameSampler = _default_sampler(),
        transform: Optional[PILVideoTransform] = None,
        frame_counter: Optional[Callable[[Path], int]] = None,
    ) -> None:

        if transform is None:
            transform = PILVideoToTensor()
        super().__init__(
            root_path, label_set=None, sampler=sampler, transform=transform
        )

        self.records = record_set

    def __getitem__(
        self, index: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Label]]:
        video_rec = self.records[index]
        video_path = os.path.join(self.root_path, video_rec.path)
        # frames_idx = self.sampler.sample(video_rec.num_frames)
        frames_idx = self.sampler.sample(_get_videofile_frame_count(video_path))
        # print(f'num_frames: {video_rec.num_frames}')
        # print(f'measured_num_frames: {_get_videofile_frame_count(video_path)}')
        # print(f'frames_idx: {frames_idx}')
        frames = self._load_frames(frames_idx, video_path)

        if self.labels is not None:
            label = self.labels[index]
        else:
            label = empty_label

        frames, label = invoke_transform(self.transform, frames, label)

        return frames, video_rec.label
        if label is empty_label:
            return frames
        print(frames, video_rec.label)
        return frames, video_rec.label

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _load_frames(
        frame_idx: Union[slice, List[slice], List[int]], video_file: Path
    ) -> Iterator[Image]:
        from torchvideo.internal.readers import default_loader

        return default_loader(video_file, frame_idx)


class VideoFolderDataset(VideoDataset):
    """Dataset stored as a folder of videos, where each video is a single example
    in the dataset.

    The folder hierarchy should look something like this: ::

        root/video1.mp4
        root/video2.mp4
        ...
    """

    def __init__(
        self,
        root_path: Union[str, Path],
        filter: Optional[Callable[[Path], bool]] = None,
        label_set: Optional[LabelSet] = None,
        sampler: FrameSampler = _default_sampler(),
        transform: Optional[PILVideoTransform] = None,
        frame_counter: Optional[Callable[[Path], int]] = None,
    ) -> None:
        """
        Args:
            root_path: Path to dataset folder on disk. The contents of this folder
                should be video files.
            filter: Optional filter callable that decides whether a given example video
                is to be included in the dataset or not.
            label_set: Optional label set for labelling examples.
            sampler: Optional sampler for drawing frames from each video.
            transform: Optional transform over the list of frames.
            frame_counter: Optional callable used to determine the number of frames
                each video contains. The callable will be passed the path to a video and
                should return a positive integer representing the number of frames.
                This tends to be useful if you've precomputed the number of frames in a
                dataset.
        """
        if transform is None:
            transform = PILVideoToTensor()
        super().__init__(
            root_path, label_set=label_set, sampler=sampler, transform=transform
        )
        self._video_paths = self._get_video_paths(self.root_path, filter)
        self.labels = self._label_examples(self._video_paths, label_set)
        self.video_lengths = self._measure_video_lengths(
            self._video_paths, frame_counter
        )

    @property
    def video_ids(self):
        return self._video_paths

    # TODO: This is very similar to ImageFolderVideoDataset consider merging into
    #  VideoDataset
    def __getitem__(self, index: int) -> Union[Any, Tuple[Any, Label]]:
        video_file = self._video_paths[index]
        video_length = self.video_lengths[index]
        frames_idx = self.sampler.sample(video_length)
        frames = self._load_frames(frames_idx, video_file)

        if self.labels is not None:
            label = self.labels[index]
        else:
            label = empty_label

        frames, label = invoke_transform(self.transform, frames, label)

        if label is empty_label:
            return frames
        return frames, label

    def __len__(self):
        return len(self._video_paths)

    @staticmethod
    def _measure_video_lengths(video_paths, frame_counter):
        if frame_counter is None:
            frame_counter = _get_videofile_frame_count
        return [frame_counter(vid_path) for vid_path in video_paths]

    @staticmethod
    def _label_examples(video_paths, label_set: Optional[LabelSet]):
        if label_set is None:
            return None
        else:
            return [label_set[video_path.name] for video_path in video_paths]

    @staticmethod
    def _get_video_paths(root_path, filter):
        return sorted(
            [
                child
                for child in root_path.iterdir()
                if _is_video_file(child) and (filter is None or filter(child))
            ]
        )

    @staticmethod
    def _load_frames(
        frame_idx: Union[slice, List[slice], List[int]], video_file: Path
    ) -> Iterator[Image]:
        from torchvideo.internal.readers import default_loader

        return default_loader(video_file, frame_idx)

