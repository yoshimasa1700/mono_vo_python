#!/usr/bin/env python
# coding: utf-8

__author__ = "Masahiko Toyoshi"
__copyright__ = "Copyright 2007, Masahiko Toyoshi."
__license__ = "GPL"
__version__ = "1.0.0"

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataset import create_dataset


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='kitti')
    parser.add_argument('--path', required=True)

    return parser.parse_args()


def main():
    options = parse_argument()
    dataset = create_dataset(options)

    feature_detector = cv2.FastFeatureDetector_create(threshold=25,
                                                      nonmaxSuppression=True)

    lk_params = dict(winSize=(21, 21),
                     criteria=(cv2.TERM_CRITERIA_EPS |
                               cv2.TERM_CRITERIA_COUNT, 30, 0.03))

    current_pos = np.zeros((3, 1))
    current_rot = np.eye(3)

    plt.gca().set_aspect('equal', adjustable='box')

    print("{} images found.".format(dataset.image_count))

    prev_image = None

    valid_ground_truth = False
    if dataset.ground_truth is not None:
        valid_ground_truth = True

    if dataset.camera_matrix is not None:
        camera_matrix = dataset.camera_matrix()
    else:
        camera_matrix = np.array([[718.8560, 0.0, 607.1928],
                                  [0.0, 718.8560, 185.2157],
                                  [0.0, 0.0, 1.0]])
    
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

        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_image,
                                               image, points,
                                               None, **lk_params)

        E, mask = cv2.findEssentialMat(p1, points, camera_matrix,
                                       cv2.RANSAC, 0.999, 1.0, None)

        hoge, R, t, mask = cv2.recoverPose(E, p1, points, camera_matrix)

        current_pos += current_rot.dot(t)
        current_rot = R.dot(current_rot)

        # get ground truth if eist.
        if valid_ground_truth:
            ground_truth = dataset.ground_truth[index]
            plt.scatter(ground_truth[0, 3],
                        ground_truth[2, 3],
                        marker='^',
                        c='r')

        plt.scatter(current_pos[0][0], current_pos[2][0])
        plt.pause(.01)

        img = cv2.drawKeypoints(image, keypoint, None)

        # cv2.imshow('image', image)
        cv2.imshow('feature', img)
        cv2.waitKey(1)

        prev_image = image
        prev_keypoint = keypoint

if __name__ == "__main__":
    main()
