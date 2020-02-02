from typing import Optional
from collections import namedtuple

VideoRecord = namedtuple('VideoRecord', ['path', 'label'])
MultiLabelVideoRecord = namedtuple('MultiLabelVideoRecord', ['path', 'labels'])
FrameFolderRecord = namedtuple('FrameFolderRecord', ['path', 'num_frames', 'label'])


class RecordSet:

    def __init__(self, metafile, sep: Optional[str] = ' '):
        self.records = []
        self.metafile = metafile
        with open(metafile) as f:
            for line in f:
                path, label = line.strip().split(sep)
                self.records.append(VideoRecord(path, int(label)))

    def __getitem__(self, idx: int) -> VideoRecord:
        return self.records[idx]

    def __len__(self):
        return len(self.records)


class MultiLabelRecordSet:

    def __init__(self, metafile,
                 category_filename,
                 sep: Optional[str] = ','):
        self.records = []
        self.metafile = metafile
        self.category_filename = category_filename

        with open(category_filename) as f:
            self.cat2label = {k: int(v) for k, v in dict(line.strip().split(',') for line in f).items()}

        with open(metafile) as f:
            for line in f:
                path, *categories = line.strip().split(sep)
                self.records.append(MultiLabelVideoRecord(path, [int(self.cat2label[cat]) for cat in categories]))

    def __getitem__(self, idx: int) -> MultiLabelVideoRecord:
        return self.records[idx]

    def __len__(self):
        return len(self.records)


class FrameRecordSet(RecordSet):

    def __init__(self, metafile, sep: Optional[str] = ' '):
        self.records = []
        self.metafile = metafile
        with open(metafile) as f:
            for line in f:
                path, num_frames, label = line.strip().split(sep)
                self.records.append(FrameFolderRecord(path + '.mp4', int(num_frames), int(label)))

    def __getitem__(self, idx: int) -> FrameFolderRecord:
        return self.records[idx]
