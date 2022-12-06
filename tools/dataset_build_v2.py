"""
save as npz. files(after compressed)
build prophesee gen4 dataset
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import os
from glob import glob
from dataloader.prophesee.src.io.psee_loader import PSEELoader
import tqdm


DELTA_T = 50000
SKIP_T = 50000 * 10  # skip the first 0.5s
HEIGHT = 720
WIDTH = 1280
TYPE = [('t', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), ('class_id', 'u1'), ('class_confidence', '<f4'), ('track_id', '<u4')]


def split_file(file_path, save_path, file_name, height, width, delta_t=None, skip=None):
    event_videos = [PSEELoader(file_path)]
    box_videos = [PSEELoader(glob(file_path.split('_td.dat')[0] + '*.npy')[0])]

    for v in event_videos + box_videos:
        v.seek_time(skip)

    idx = 0
    npz_e, npz_l = [], []
    while not sum([video.done for video in event_videos]):
        boxes_filter = []
        track_id = []
        events = [video.load_delta_t(delta_t) for video in event_videos]
        boxes = [video.load_delta_t(delta_t) for video in box_videos]

        for box in boxes[0]:
            if box["track_id"] not in track_id:
                boxes_filter.append(box)  # filter same object id in each 50ms
                track_id.append(box["track_id"])

        boxes = [np.array(boxes_filter).astype(TYPE)]

        save_event_dir = os.path.join(save_path, "events", file_name[:-7])
        save_box_dir = os.path.join(save_path, "labels", file_name[:-7])

        event_file_name = "ev" + ("%03d" % (idx // 10)) + ".npz"
        box_file_name = "lb" + ("%03d" % (idx // 10)) + ".npz"

        save_event_file = os.path.join(save_event_dir, event_file_name)
        save_box_file = os.path.join(save_box_dir, box_file_name)

        if events[0].shape[0] != 0:
            npz_e.append(events[0])
            npz_l.append(boxes[0])
        else:
            print("\033[0;33m Sequence %d [%s] has no event streams! \033[0m" % (idx, file_name))

        # save .npz file if not empty
        if len(npz_e) == 10:
            if not (os.path.exists(save_event_dir) and os.path.exists(save_box_dir)):
                os.mkdir(save_event_dir)
                os.mkdir(save_box_dir)
            np.savez_compressed(save_event_file, e0=npz_e[0], e1=npz_e[1], e2=npz_e[2], e3=npz_e[3], e4=npz_e[4],
                     e5=npz_e[5], e6=npz_e[6], e7=npz_e[7], e8=npz_e[8], e9=npz_e[9])
            np.savez_compressed(save_box_file, l0=npz_l[0], l1=npz_l[1], l2=npz_l[2], l3=npz_l[3], l4=npz_l[4],
                     l5=npz_l[5], l6=npz_l[6], l7=npz_l[7], l8=npz_l[8], l9=npz_l[9])
            npz_e, npz_l = [], []  # initialize

        idx += 1


if __name__ == "__main__":
    dataset_dir = "/home/wds/Desktop/testfilelist00/test"
    save_dir = "/home/wds/Desktop/prophesee_gen4_npz/test/testfilelist00"
    files = os.listdir(dataset_dir)
    files = [time_seq_name for time_seq_name in files if time_seq_name[-3:] == 'dat']

    print("\033[0;31mStarting to splitting the dataset! \033[0m")
    pbar = tqdm.tqdm(total=len(files), unit="File", unit_scale=True)
    for file in files:
        abs_path = os.path.join(dataset_dir, file)
        # skip the first 0.5s
        split_file(abs_path, save_dir, file, height=HEIGHT, width=WIDTH, delta_t=DELTA_T, skip=SKIP_T)
        pbar.update()
    pbar.close()
    print("\033[0;31mDataset is already split! \033[0m")
