from typing import Any
import numpy as np
import cv2
# import rospy
import math
import matplotlib.pyplot as plt
import PIL.Image as pil_image
import torch
import warnings
import argparse
import json
import sched, time
from pathlib import Path
import sys
from tqdm import tqdm

if __name__ == '__main__':
    camera_pose_path = "/home/srihari/tcs/data/mango_videos/GOPR1731_reconstruction/model/images.txt"
    points3D_path = "/home/srihari/tcs/data/mango_videos/GOPR1731_reconstruction/model/points3D.txt"
    detections_path = "/home/srihari/tcs/data/mango_videos/GOPR1731_reconstruction/detections.txt"
    wc_path = "/home/srihari/tcs/data/mango_videos/GOPR1731_reconstruction/wcs.txt"
    frame_num = 844

    mango_detections = []
    with open(detections_path, 'r') as f:   
        for i in f.readlines():
            if i.strip() == '':
                mango_detections.append([])
                continue
            bboxes = i.strip().split(';')
            bboxes = [[float(x) for x in y.split(',')] for y in bboxes]
            mango_detections.append(bboxes)

    camera_poses = [[] for i in range(frame_num)]
    tracked_points = [[] for i in range(frame_num)]
    with open(camera_pose_path, 'r') as f:
        check = 0
        for l in f.readlines():
            if l[0] == '#':
                continue
            if check == 0:
                l = l.strip().split()
                img_file_name = l[-1]
                idx = int(img_file_name[5:-4])-1
                # idx = int(img_file_name[:-4])-1
                pose = [float(i) for i in l[1:8]]
                camera_poses[idx] = [pose[0:4], pose[4:7]]
            else:
                l = l.strip().split()
                points2D = []
                x,y = 0,0
                check_1 = 0
                for i in l:
                    if check_1 == 0:
                        x = float(i)
                    elif check_1 == 1:
                        y = float(i)
                    elif i != '-1':
                        points2D.append((x,y,int(i)))
                    check_1 += 1
                    if check_1 == 3:
                        check_1 = 0
                tracked_points[idx] = points2D
            check = 1-check

    points3D = {}
    with open(points3D_path, 'r') as f:
        for l in f.readlines():
            if l[0] == '#':
                continue
            l = l.strip().split()
            idx = int(l[0])
            points = [float(i) for i in l[1:4]]
            points3D[idx] = points

    print("Starting...")

    with open(wc_path, 'w') as f:
        for frame_counter, bboxes in tqdm(enumerate(mango_detections)):
            bboxes_mps = [np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]) for bbox in bboxes]
            points2D = tracked_points[frame_counter]
            points2D = np.array(points2D, dtype=np.float32)
            foreign_keys = points2D[:,-1]
            points2D = points2D[:,:-1]
            # finding the closest pixel to each bbox
            closest_foreign_keys = []
            closest_points2D = []
            for bbox_mp, bbox in zip(bboxes_mps,bboxes):
                good_idxs = np.logical_and(np.logical_and(bbox[0] <= points2D[:,0], points2D[:,0] <= bbox[2]), np.logical_and(bbox[1] <= points2D[:,1], points2D[:,1] <= bbox[3]))
                points2D_ = points2D[good_idxs]
                foreign_keys_ = foreign_keys[good_idxs]
                if len(points2D_):                
                    dist_2 = np.sum((points2D_-bbox_mp)**2, axis=1)
                    cfk = np.argmin(dist_2)
                    closest_foreign_keys.append(foreign_keys_[cfk])
                    closest_points2D.append(points2D_[cfk])
                else:
                    closest_foreign_keys.append(-1)

            '''Step:2'''
            wcs = []
            for cfk in closest_foreign_keys:
                if cfk == -1:
                    wcs.append([-1,-1,-1])
                else:
                    wcs.append(points3D[cfk])

            s = ""
            for i in wcs:
                s += f"{i[0]},{i[1]},{i[2]};"
            s = s[:-1] + '\n'

            f.write(s)