"""
small executable to show the content of the Prophesee dataset
Copyright: (c) 2019-2020 Prophesee
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('./')

import os
import numpy as np
import cv2
import argparse
from dataloader.prophesee.src.visualize import vis_utils as vis


def play_files_parallel(td_dir, npy_files, height, width):
    """
    Plays simultaneously files and their boxes in a rectangular format.
    """
    filelist_dirs = os.listdir(td_dir)
    ev_filelist = list()
    lb_filelist = list()
    for td in filelist_dirs:
        ev_path = os.path.join(td_dir, td, "events")
        lb_path = os.path.join(td_dir, td, "labels")

        for pth in sorted(os.listdir(ev_path)):
            ev_dir_path = os.path.join(ev_path, pth)
            lb_dir_path = os.path.join(lb_path, pth)

            for ev, lb in zip(sorted(os.listdir(ev_dir_path)), sorted(os.listdir(lb_dir_path))):
                ev_root = os.path.join(ev_dir_path, ev)
                lb_root = os.path.join(lb_dir_path, lb)

                ev_filelist.append(ev_root)
                lb_filelist.append(lb_root)

    # open the video object for the input files
    p_idx = 0
    boxes = np.load(npy_files, allow_pickle=True)  # load detection results
    for idx in range(len(ev_filelist)):
        # use the naming pattern to find the corresponding box file
        labelmap = vis.LABELMAP if height == 240 else vis.LABELMAP_LARGE

        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.namedWindow('prediction', cv2.WINDOW_NORMAL)

        # load events and boxes from all files
        events = np.load(ev_filelist[idx])

        for npz_num in range(len(events)):
            ev_npz = "e" + str(npz_num)
            events_ = events[ev_npz]
            boxes_ = boxes[p_idx]

            im = frame[0:height, 0:width]
            # call the visualization functions
            im = vis.make_binary_histo(events_, img=im, width=width, height=height)
            vis.drawing_bboxes(im, boxes_, labelmap=labelmap)
            p_idx += 1
            # display the result
            cv2.imshow('prediction', frame)
            cv2.waitKey(50)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description="visualize one or several event files along with their boxes")
    parser.add_argument(
        '-t', "--td_dir",
        default="/home/wds/Desktop/exp_prophesee/train",
        type=str, help="input event files, annotation files are expected to be in the same folder")
    parser.add_argument(
        '-n', "--npy_file", default="/home/wds/Desktop/DMANet/det_result/prediction.npy", type=str,
        help="model predictions(bounding boxes), type [x_min, y_min, x_max, y_max, confidence, cls_id]")
    parser.add_argument('--height', default=720, type=int, help="image height")
    parser.add_argument('--width', default=1280, type=int, help="image width")

    return parser.parse_args()


if __name__ == "__main__":
    ARGS = parse_args()
    play_files_parallel(ARGS.td_dir, ARGS.npy_file, ARGS.height, ARGS.width)
