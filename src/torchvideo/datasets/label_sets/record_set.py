from typing import Optional
from collections import namedtuple

VideoRecord = namedtuple('VideoRecord', ['path', 'label'])
FrameFolderRecord = namedtuple('FrameFolderRecord', ['path', 'num_frames', 'label'])


class RecordSet:

    def __init__(self, filename, sep: Optional[str] = ' '):
        self.records = []
        self.filename = filename
        with open(filename) as f:
            for line in f:
                path, label = line.strip().split(sep)
                self.records.append(VideoRecord(path, int(label)))

    def __getitem__(self, idx: int) -> VideoRecord:
        return self.records[idx]

    def __len__(self):
        return len(self.records)


class FrameRecordSet(RecordSet):

    def __init__(self, filename, sep: Optional[str] = ' '):
        self.records = []
        self.filename = filename
        with open(filename) as f:
            for line in f:
                path, num_frames, label = line.strip().split(sep)
                self.records.append(FrameFolderRecord(path + '.mp4', int(num_frames), int(label)))

    def __getitem__(self, idx: int) -> FrameFolderRecord:
        return self.records[idx]

