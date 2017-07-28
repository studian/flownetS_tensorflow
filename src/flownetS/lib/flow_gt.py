#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''According to the Matlab source code of Deqing Sun
Contact: dqsun@cs.brown.edu

Author: Song Wang, University of Science and Technology of China
Contact: swang926@mail.ustc.edu.cn
Date: 2017-02-24 02:51 '''

from __future__ import division
import struct
import numpy as np
import cv2


class Flow(object):
    def __init__(self, floname):
        self.floname = floname

    def read_flo(self):
            with open(self.floname, "rb") as f:
                data = f.read()
            self.width = struct.unpack('@i', data[4:8])[0]
            self.height = struct.unpack('@i', data[8:12])[0]
            self.flodata = np.zeros((self.height, self.width, 2))
            for i in range(self.width*self.height):
                data_u = struct.unpack('@f', data[12+8*i:16+8*i])[0]
                data_v = struct.unpack('@f', data[16+8*i:20+8*i])[0]
                n = int(i / self.width)
                k = np.mod(i, self.width)
                self.flodata[n, k, :] = [data_u, data_v]
            return self.flodata

    def get_colorwheel(self):
        RY = 15
        YG = 6
        GC = 4
        CB = 11
        BM = 13
        MR = 6
        ncols = RY + YG + GC + CB + BM + MR
        colorwheel = np.zeros([ncols, 3])

        col = 0
        # RY
        colorwheel[0:RY, 0] = 255
        colorwheel[0:RY, 1] = np.floor([x * (255/RY) for x in range(RY)])
        col = col + RY
        # YG
        colorwheel[col:col+YG, 0] = 255 - np.floor([x * (255/YG) for x in range(YG)])
        colorwheel[col:col+YG, 1] = 255
        col = col + YG
        # GC
        colorwheel[col:col+GC, 1] = 255
        colorwheel[col:col+GC, 2] = np.floor([x * (255/GC) for x in range(GC)])
        col = col + GC
        # CB
        colorwheel[col:col+CB, 1] = 255 - np.floor([x * (255/CB) for x in range(CB)])
        colorwheel[col:col+CB, 2] = 255
        col = col + CB
        # BM
        colorwheel[col:col+BM, 2] = 255
        colorwheel[col:col+BM, 0] = np.floor([x * (255/BM) for x in range(BM)])
        col = col + BM
        # MR
        colorwheel[col:col+MR, 2] = 255 - np.floor([x * (255/MR) for x in range(MR)])
        colorwheel[col:col+MR, 0] = 255
        return colorwheel

    def print_flo(self):
        self.flodata = self.read_flo()
        u = self.flodata[:, :, 0]
        v = self.flodata[:, :, 1]
        img = np.zeros([self.height, self.width, 3])
        # normalization
        rad = np.amax((u ** 2 + v ** 2) ** 0.5)
        eps = np.finfo(float).eps
        u = u / (rad + eps)
        v = v / (rad + eps)
        # image a colorwheel, if we have arc length and radius, it's easy to locate an exact color
        colorwheel = self.get_colorwheel()
        rad = (u ** 2 + v ** 2) ** 0.5
        arc = np.arctan2(-v, -u) / np.pi
        # the number of color's level in which R/G/B channel
        ncols = colorwheel.shape[0]
        # [-1, 1] maped to [1, ncols]
        level = (arc+1) / 2 * (ncols-1) + 1
        level = level.reshape((-1, 1))
        level_floor = [int(x) for x in level]
        level_ceil = [x+1 for x in level_floor]
        for x in level_ceil:
            if x == ncols + 1:
                x = 1
        mask = list(map(lambda x: x[0]-x[1], zip(level, level_floor)))
        for i in range(colorwheel.shape[1]):
            tmp = colorwheel[:, i]
            tmp = list(tmp)
            col0 = []
            col1 = []
            for x in level_floor:
                col0.append(tmp[x-1])

            for x in level_ceil:
                col1.append(tmp[x-1])
            # transfer to matrix for compute
            mask = np.array(mask).reshape((self.height, self.width))
            col0 = np.array(col0).reshape((self.height, self.width))
            col0 = col0 / 255.
            col1 = np.array(col1).reshape((self.height, self.width))
            col1 = col1 / 255.
            col = (1-mask) * col0 + mask * col1
            # dont konw why need follow code,
            # increase saturation with radius?
            col = col.reshape((-1, 1))
            rad = rad.reshape((-1, 1))
            m = 0
            for x in rad:
                if x <= 1:
                    col[m][0] = 1 - x * (1 - col[m][0])
                    col[m][0] = int(255 * col[m][0])
                    m = m + 1
                else:
                    col[m][0] = col[m][0] * 0.75
                    col[m][0] = int(255 * col[m][0])
                    m = m + 1
            col = col.reshape((self.height, self.width))
            img[:, :, i] = col
        return img


flo_file = Flow('gt.flo').read_flo()
image = Flow('gt.flo').print_flo()
cv2.imshow('image', image.astype(np.uint8))
cv2.imwrite('gt.png', image)
cv2.waitKey()
