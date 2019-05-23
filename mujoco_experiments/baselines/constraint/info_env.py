import numpy as np
from skimage.measure import label, regionprops
"""
Function takes in as input the env.step as a numpy array
For example when stepping through the env;
frame = np.asarray(env.step(action))
Then pass frame to the identify_mother function
which returns a boolean true or false and bounding
box coordinates.
"""


def identify_mothership(obs):
    """ Takes an observation and returns whether the mothership is on the
    screen and a bounding box.

    Args:
        obs: observation returned from a SpaceInvaders environment

    Returns:
        (bool, int 4 tuple) which indicate whether the mothership is on the
        screen and where respectively.
    """
    frames = np.asarray(obs)
    sliced_frames = np.squeeze(
        frames[8:25, :, 1:2])  # magic numbers concocted by Chris the wizard
    label_image = label(sliced_frames, return_num=True, connectivity=2)
    if label_image[1] == 1:
        region = regionprops(label_image[0])
        box = region[0].bbox  #min_row, min_col, max_row, max_col
        return True, box
    else:
        return False, (0, 0, 0, 0)
