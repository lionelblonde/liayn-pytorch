import os.path as osp
import hashlib
import time
import cv2
import numpy as np


def record_video(save_dir, index, obs):
    """Record a video from samples collected at evalutation time."""
    # Unstack the frames if stacked, while leaving colors unaltered
    frames = np.split(obs, 1, axis=-1)
    frames = np.concatenate(np.array(frames), axis=0)
    frames = [np.squeeze(a, axis=0)
              for a in np.split(frames, frames.shape[0], axis=0)]

    # Create OpenCV video writer
    vname = "render-{}".format(index)
    frameSize = (obs.shape[-2],
                 obs.shape[-3])
    writer = cv2.VideoWriter(filename="{}.mp4".format(osp.join(save_dir, vname)),
                             fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                             fps=25,
                             frameSize=frameSize,
                             isColor=True)

    for frame in frames:
        # Add frame to video
        writer.write(frame)
    writer.release()
    cv2.destroyAllWindows()
    # Delete the object
    del frames


class OpenCVImageViewer(object):
    """Viewer used to render simulations."""

    def __init__(self, q_to_exit=True):
        self._q_to_exit = q_to_exit
        # Create unique identifier
        hash_ = hashlib.sha1()
        hash_.update(str(time.time()).encode('utf-8'))
        # Create window
        self._window_name = str(hash_.hexdigest()[:20])
        cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)
        self._isopen = True

    def __del__(self):
        cv2.destroyWindow(self._window_name)
        self._isopen = False

    def imshow(self, img):
        # Convert image to BGR format
        cv2.imshow(self._window_name, img[:, :, [2, 1, 0]])
        # Listen for escape key, then exit if pressed
        if cv2.waitKey(1) == ord('q') and self._q_to_exit:
            exit()

    @property
    def isopen(self):
        return self._isopen

    def close(self):
        pass
