import argparse
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='kitti')
    parser.add_argument('--path', required=True)

    return parser.parse_args()


class CameraParameters():
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    @property
    def cameraMatrix(self):
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

    def loadCameraParameters(self, calibfile):
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


def main():
    options = parseArguments()
    dataset = dataset_dict[options.dataset](options.path)

    feature_detector = cv2.FastFeatureDetector_create(threshold=25,
                                                      nonmaxSuppression=True)

    lk_params = dict(winSize=(21, 21),
                     criteria=(cv2.TERM_CRITERIA_EPS |
                               cv2.TERM_CRITERIA_COUNT, 30, 0.03))

    current_pos = np.zeros((3, 1))
    current_rot = np.eye(3)

    plt.gca().set_aspect('equal', adjustable='box')

    print(dataset.image_count)

    prev_image = None
    for index in xrange(dataset.image_count):
        # load image
        image = cv2.imread(dataset.image_path_left(index))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # main process
        keypoint = feature_detector.detect(image, None)

        if prev_image is None:
            prev_image = image
            prev_keypoint = keypoint
            continue

        points = np.array(map(lambda x: [x.pt], prev_keypoint),
                          dtype=np.float32)

        # p0 = cv2.goodFeaturesToTrack(prev_image, mask=None, **feature_params)

        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_image,
                                               image, points,
                                               None, **lk_params)

        cameraMatrix = np.array([[718.8560, 0.0, 607.1928],
                                 [0.0, 718.8560, 185.2157],
                                 [0.0, 0.0, 1.0]])
        E, mask = cv2.findEssentialMat(p1, points, cameraMatrix, cv2.RANSAC,
                                       0.999, 1.0, None)

        hoge, R, t, mask = cv2.recoverPose(E, p1, points, cameraMatrix)

        current_pos += current_rot.dot(t)
        current_rot = R.dot(current_rot)

        # print(E)
        # print(t)
        # print(R)
        # print(current_pos[0][0])
        # print(current_pos)

        plt.scatter(current_pos[0][0], current_pos[2][0])
        plt.pause(.01)

        img = cv2.drawKeypoints(image, keypoint, None)

        cv2.imshow('image', image)
        cv2.imshow('feature', img)
        cv2.waitKey(1)

        prev_image = image
        prev_keypoint = keypoint

if __name__ == "__main__":
    main()
