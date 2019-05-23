from skimage.measure import label, regionprops
import numpy as np

def identify_mothership(obs):
    """ Takes an observation and returns whether the mothership is on the
    screen and a bounding box.

    Args:
        obs: observation returned from a SpaceInvaders environment

    Returns:
        list of a bounding box for each frame
    """
    frames = np.asarray(obs[0])
    if len(frames.shape) == 2:
        frame_region = np.squeeze(frames[8:25, :])
        label_ = label(frame_region, return_num=True, connectivity=2)
        boxes = regionprops(label_[0])[0].bbox if label_[1] == 1 else (0,0,0,0)
    if len(frames.shape) == 3:
        frame_regions = [np.squeeze(frames[8:25, :, i]) for i in range(frames.shape[2])]
        labels = [label(x, return_num=True, connectivity=2) for x in frame_regions]
        boxes = [regionprops(x[0])[0].bbox if x[1] == 1 else (0,0,0,0) for x in labels]
    return boxes
