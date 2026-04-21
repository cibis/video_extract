from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Callable
import tensorflow as tf


class VideoExtractBase(ABC):
    """
    Abstract base class defining the interface for VideoExtract implementations.
    """

    @abstractmethod
    def __init__(self,
                 detector,
                 dtype: tf.dtypes.DType,
                 detected_object_classes: List[str],
                 minimum_detection_score: float,
                 source_video: str,
                 result_video: str,
                 inspection_rate_in_seconds: float,
                 maximum_missed_detections_time_in_seconds: float,
                 minimum_video_part_size_in_seconds: float,
                 video_tmp_output_path: str,
                 normalize_results: Optional[Callable] = None):
        raise NotImplementedError

    @abstractmethod
    def get_video_duration(self) -> Tuple[int, int]:
        """Return (duration_seconds, fps)"""
        raise NotImplementedError

    @abstractmethod
    def get_video_part_length(self, part_num: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_occurrences(self, part_num: int, part_index: int) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def extract_occurrences(self, occurrences: List[int]) -> int:
        raise NotImplementedError

    @abstractmethod
    def extract_all_occurrences(self) -> int:
        raise NotImplementedError
