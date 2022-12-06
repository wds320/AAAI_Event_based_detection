"""
This code aims to check dataset after split.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import numpy as np
import cv2
import argparse
from dataloader.prophesee.src.visualize import vis_utils as vis
from numpy.lib import recfunctions as rfn


def cropToFrame(np_bbox, height, width):
    """Checks if bounding boxes are inside frame. If not crop to border"""
    boxes = []
    for box in np_bbox:
        if box[2] > 1280:  # filter error label
            continue

        if box[0] < 0:  # x < 0 & w > 0
            box[2] += box[0]
            box[0] = 0
        if box[1] < 0:  # y < 0 & h > 0
            box[3] += box[1]
            box[1] = 0
        if box[0] + box[2] > width:  # x+w>1280
            box[2] = width - box[0]
        if box[1] + box[3] > height:  # y+h>720
            box[3] = height - box[1]

        if box[2] > 0 and box[3] > 0 and box[0] < width and box[1] < height:
            boxes.append(box)
    boxes = np.array(boxes).reshape(-1, 5)
    return boxes


def filter_boxes(boxes, min_box_diag=60, min_box_side=20):
    """Filters boxes according to the paper rule.
    To note: the default represents our threshold when evaluating GEN4 resolution (1280x720)
    To note: we assume the initial time of the video is always 0
    :param boxes: (np.ndarray)
                 structured box array with fields ['t','x','y','w','h','class_id','track_id','class_confidence']
                 (example BBOX_DTYPE is provided in src/box_loading.py)
    Returns:
        boxes: filtered boxes
    """
    width = boxes[:, 2]
    height = boxes[:, 3]
    diag_square = width ** 2 + height ** 2
    mask = (diag_square >= min_box_diag ** 2) * (width >= min_box_side) * (height >= min_box_side)
    return boxes[mask]


def draw_bboxes(img, boxes, labelmap):
    """
    draw bboxes in the image img
    """
    colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)
    colors = [tuple(*item) for item in colors.tolist()]
    for i in range(boxes.shape[0]):
        pt1 = (int(boxes[i][0]), int(boxes[i][1]))
        size = (int(boxes[i][2]), int(boxes[i][3]))
        pt2 = (pt1[0] + size[0], pt1[1] + size[1])
        class_id = int(boxes[i][4])
        class_name = labelmap[class_id % len(labelmap)]
        color = colors[class_id * 60 % 255]
        center = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.rectangle(img, pt1, pt2, color, 2)
        cv2.putText(img, class_name, (center[0], pt2[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)


def play_files_parallel(root_dir, height, width):
    """
    Plays simultaneously files and their boxes in a rectangular format.
    """
    event_dir = os.path.join(root_dir, "events")
    label_dir = os.path.join(root_dir, "labels")

    event_files = sorted(os.listdir(event_dir))
    label_files = sorted(os.listdir(label_dir))

    event_list, label_list = [], []
    for ev, lb in zip(event_files, label_files):
        event_path = os.path.join(event_dir, ev)
        label_path = os.path.join(label_dir, lb)
        for e, l in zip(sorted(os.listdir(event_path)), sorted(os.listdir(label_path))):
            event_path_ = os.path.join(event_path, e)
            label_path_ = os.path.join(label_path, l)
            event_list.append(event_path_), label_list.append(label_path_)

    assert len(event_list) == len(label_list)
    for idx in range(len(event_list)):
        # use the naming pattern to find the corresponding box file
        labelmap = vis.LABELMAP if height == 240 else vis.LABELMAP_LARGE

        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.namedWindow('gt', cv2.WINDOW_NORMAL)

        # load events and boxes from all files
        events = np.load(event_list[idx])
        boxes = np.load(label_list[idx])

        for npz_num in range(len(events)):
            ev_npz = "e" + str(npz_num)
            lb_npz = "l" + str(npz_num)
            events_ = events[ev_npz]
            boxes_ = boxes[lb_npz]
            boxes_ = rfn.structured_to_unstructured(boxes_)[:, [1, 2, 3, 4, 5]]  # (x, y, w, h, class_id)
            boxes_ = cropToFrame(boxes_, height, width)
            boxes_ = filter_boxes(boxes_, 60, 20)  # filter boxes
            im = frame[0:height, 0:width]
            # call the visualization functions
            im = vis.make_binary_histo(events_, img=im, width=width, height=height)
            draw_bboxes(im, boxes_, labelmap=labelmap)

            # display the result
            cv2.imshow('gt', frame)
            cv2.waitKey(1)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument(
        '-r', '--records', type=str,
        default="/home/wds/Desktop/prophesee_dlut/train/trainfilelist01",
        help='input event files, annotation files are expected to be in the same folder')
    parser.add_argument('--height', default=720, type=int, help="image height")
    parser.add_argument('--width', default=1280, type=int, help="image width")

    return parser.parse_args()


if __name__ == '__main__':
    ARGS = parse_args()
    play_files_parallel(ARGS.records, ARGS.height, ARGS.width)
