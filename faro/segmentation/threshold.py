import numpy as np
from faro.segmentation.base import Segmentator, remove_small_objects
import skimage

class SegmentatorThreshold(Segmentator):
    """
    Simple threshold-based segmentator.

    Normalizes the image to [0, 1] and applies a fixed intensity threshold
    (default 0.5) to create a binary mask. Connected components in the mask
    are then labelled as individual objects.
    """

    def __init__(self, threshold: float = 0.5, min_size: int = 0):
        """
        Initialize the SegmentatorThreshold object.

        Parameters:
        threshold (float): Intensity threshold on the normalized image. Defaults to 0.5.
        min_size (int): Minimum object size in pixels. Objects smaller than this are removed.
                        If 0, no filtering is performed. Defaults to 0.
        """
        self.threshold = threshold
        self.min_size = min_size

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment by thresholding the normalized image and labelling connected components.
        """
        # Normalize to [0, 1]
        img_min = image.min()
        img_max = image.max()
        if img_max - img_min > 0:
            img_normed = (image - img_min) / (img_max - img_min)
        else:
            img_normed = np.zeros_like(image, dtype=float)
        img_normed[img_normed<0] = 0
        img_normed[img_normed>1] = 1

        # Threshold and label
        binary = img_normed > self.threshold
        labels = skimage.measure.label(binary)

        if self.min_size > 0:
            # Use faro's shim, not skimage.morphology directly: scikit-image
            # 0.26 renamed the size argument and flipped its comparison.
            labels = remove_small_objects(
                labels, min_size=self.min_size, connectivity=1
            )
        return labels
