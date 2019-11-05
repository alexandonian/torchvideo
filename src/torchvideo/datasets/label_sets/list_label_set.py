from typing import Optional
from collections import namedtuple


VideoRecord = namedtuple('VideoRecord', ['path', 'num_frames', 'label'])


class RecordSet(object):
    """Label set from a list of video files and targets.

    """

    def __init__(self, filename, sep: Optional[str] = ' '):
        """

        Args:
            df: pandas DataFrame or Series containing video names/ids and their
                corresponding labels.
            col: The column to read the label from when df is a DataFrame.
        """
        self.records = []
        self.filename = filename
        with open(filename) as f:
            for line in f:
                path, num_frames, label = line.strip().split(sep)
                self.records.append(VideoRecord(path, int(num_frames), int(label)))

    def __getitem__(self, idx) -> VideoRecord:
        return self.records[idx]
