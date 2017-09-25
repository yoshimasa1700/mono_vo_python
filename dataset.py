import os
import glob
import numpy as np


class CameraParameters():
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    @property
    def camera_matrix(self):
        matrix = np.array([[self.fx, 0.0, self.cx],
                           [0.0, self.fx, self.cy],
                           [0.0, 0.0, 1.0]])
        return matrix

    def __call__(self):
        return self.cameraMatrix


class Dataset():
    def __init__(self):
        pass

    def image_path_left(self, index):
        return os.path.join(self.path, self.image_format_left).format(index)

    def count_image(self):
        extension = os.path.splitext(self.image_format_left)[-1]
        wildcard = os.path.join(self.path, '*' + extension)
        self.image_count = len(glob.glob(wildcard))

    def load_camera_parameters(self, calibfile):
        with open(calibfile, 'r') as f:
            line = f.readline()
            part = line.split()
            param = CameraParameters(part[1], part[6], part[3], part[7])

            return param


class KittiDataset(Dataset):
    def __init__(self, path):
        self.image_format_left = '{:06d}.png'
        self.path = path
        self.calibfile = os.path.join(self.path, 'calib.txt')
        self.count_image()


dataset_dict = {'kitti': KittiDataset}


def create_dataset(options):
    return dataset_dict[options.dataset](options.path)
