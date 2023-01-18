#!/usr/bin/env python
import os

import numpy as np
import rawpy
import torch
from torch.utils import data
import time
import cv2
import imageio

class RAW_Edge(data.Dataset):

    def __init__(self, root, image_list_file=None, image_list=None):
        """
        :param root: dataset directory
        :param image_list_file: contains image file names under root
        :param patch_size: if None, full images are returned, otherwise patches are returned
        :param split: train or valid
        :param upper: max number of image used for debug
        """
        assert os.path.exists(root), "root: {} not found.".format(root)
        self.root = root

        if image_list:
            self.img_files = image_list
        else:
            assert os.path.exists(image_list_file), "image_list_file: {} not found.".format(image_list_file)
            self.image_list_file = image_list_file

            self.img_files = []
            
            with open(self.image_list_file, 'r') as f:
                for img_file in f:
                    img_file = img_file.strip()
                    self.img_files.append(img_file)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_file = self.img_files[index]

        if img_file.endswith("ARW") or img_file.endswith("CR2") or img_file.endswith("raw_10") or img_file.endswith("RAF"):
            # setting 1:
            # raw = np.fromfile(os.path.join(self.root, img_file), dtype=np.uint8)
            # # raw = np.reshape(raw,(1080,1920,1))
            # raw = np.reshape(raw,(1080,1920,1))
            # # import pdb; pdb.set_trace()
            # imageio.imsave(os.path.join(self.root, img_file[:-6]+'tif'), raw)
            # # print(raw1[:10,0,0]) # [85 39 39 40 41 38 39 39 38 38]

            # raw = imageio.imread(os.path.join(self.root, img_file[:-6]+'tif'))
            # raw = raw.astype(np.float32)
            # import pdb;pdb.set_trace()
            # setting 2
            raw = rawpy.imread(os.path.join(self.root, img_file))
            raw = raw.raw_image_visible.astype(np.float32)
            
        elif img_file.endswith("npy"):
            # import pdb;pdb.set_trace()
            raw = np.load(os.path.join(self.root, img_file))
            # clear low 2 bits
            # raw &= 0xfff0 
            # 1111 1111 1111 1111
            # 0000 0000 1111 1111
            # 1111 1111 1111 0000
        elif img_file.endswith("png"):
            raw = cv2.imread(os.path.join(self.root, img_file))
        else:
            print(img_file)
            raise NotImplementedError()

        # import pdb;pdb.set_trace()
        input_full = self.pack_raw(raw)
        input_full = input_full.transpose(2, 0, 1)  # C x H x W
        input_full = np.minimum(input_full, 1.0)
        input_full = torch.from_numpy(input_full).float()

        return input_full, img_file

class Sony(RAW_Edge):
    def __init__(self, root, image_list_file, image_list):
        super(Sony, self).__init__(root, image_list_file, image_list)

    def pack_raw(self, raw):
        # pack Bayer image to 4 channels
        im = raw.astype(np.float32)
        im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level
        if np.average(im) < 0.1:
            im *= (0.1 / np.average(im)) # scale image

        im = np.expand_dims(im, axis=2)
        img_shape = im.shape
        H = img_shape[0]
        W = img_shape[1]

        out = np.concatenate((im[0:H:2, 0:W:2, :],
                              im[0:H:2, 1:W:2, :],
                              im[1:H:2, 1:W:2, :],
                              im[1:H:2, 0:W:2, :]), axis=2)
        return out

class Fuji(RAW_Edge):
    def __init__(self, root, image_list_file, image_list):
        super(Fuji, self).__init__(root, image_list_file, image_list)

    def pack_raw(self, raw):
        # pack X-Trans image to 9 channels
        im = raw.astype(np.float32)
        im = np.maximum(im - 1024, 0) / (16383 - 1024)  # subtract the black level
        if np.average(im) < 0.1:
            im *= (0.1 / np.average(im)) # scale image

        img_shape = im.shape
        H = (img_shape[0] // 6) * 6
        W = (img_shape[1] // 6) * 6

        out = np.zeros((H // 3, W // 3, 9))

        # 0 R
        out[0::2, 0::2, 0] = im[0:H:6, 0:W:6]
        out[0::2, 1::2, 0] = im[0:H:6, 4:W:6]
        out[1::2, 0::2, 0] = im[3:H:6, 1:W:6]
        out[1::2, 1::2, 0] = im[3:H:6, 3:W:6]

        # 1 G
        out[0::2, 0::2, 1] = im[0:H:6, 2:W:6]
        out[0::2, 1::2, 1] = im[0:H:6, 5:W:6]
        out[1::2, 0::2, 1] = im[3:H:6, 2:W:6]
        out[1::2, 1::2, 1] = im[3:H:6, 5:W:6]

        # 1 B
        out[0::2, 0::2, 2] = im[0:H:6, 1:W:6]
        out[0::2, 1::2, 2] = im[0:H:6, 3:W:6]
        out[1::2, 0::2, 2] = im[3:H:6, 0:W:6]
        out[1::2, 1::2, 2] = im[3:H:6, 4:W:6]

        # 4 R
        out[0::2, 0::2, 3] = im[1:H:6, 2:W:6]
        out[0::2, 1::2, 3] = im[2:H:6, 5:W:6]
        out[1::2, 0::2, 3] = im[5:H:6, 2:W:6]
        out[1::2, 1::2, 3] = im[4:H:6, 5:W:6]

        # 5 B
        out[0::2, 0::2, 4] = im[2:H:6, 2:W:6]
        out[0::2, 1::2, 4] = im[1:H:6, 5:W:6]
        out[1::2, 0::2, 4] = im[4:H:6, 2:W:6]
        out[1::2, 1::2, 4] = im[5:H:6, 5:W:6]

        out[:, :, 5] = im[1:H:3, 0:W:3]
        out[:, :, 6] = im[1:H:3, 1:W:3]
        out[:, :, 7] = im[2:H:3, 0:W:3]
        out[:, :, 8] = im[2:H:3, 1:W:3]

        return out


class Canon(RAW_Edge):
    def __init__(self, root, image_list_file, image_list):
        super(Canon, self).__init__(root, image_list_file, image_list)

    def pack_raw(self, raw):
        # pack Bayer image to 4 channels
        im = raw.astype(np.float32)
        im = np.maximum(im - 2048, 0) / (16383 - 2048)  # subtract the black level
        if np.average(im) < 0.1:
            im *= (0.1 / np.average(im)) # scale image

        im = np.expand_dims(im, axis=2)
        img_shape = im.shape
        H = img_shape[0]
        W = img_shape[1]

        out = np.concatenate((im[0:H:2, 0:W:2, :],
                              im[0:H:2, 1:W:2, :],
                              im[1:H:2, 1:W:2, :],
                              im[1:H:2, 0:W:2, :]), axis=2)

        return out


class GC4653(RAW_Edge):
    def __init__(self, root, image_list_file, image_list):
        super(GC4653, self).__init__(root, image_list_file, image_list)

    def pack_raw(self, raw):
        # pack Bayer image to 4 channels
        
        im = raw.astype(np.float32)
        im = np.maximum(im - 64, 0) / (1023 - 64)  # subtract the black level

        if np.average(im) < 0.1:
            im *= (0.1 / np.average(im)) # scale image

        im = np.expand_dims(im, axis=2)
        img_shape = im.shape
        H = img_shape[0]
        W = img_shape[1]

        out = np.concatenate((im[0:H:2, 0:W:2, :],
                              im[0:H:2, 1:W:2, :],
                              im[1:H:2, 1:W:2, :],
                              im[1:H:2, 0:W:2, :]), axis=2)
        return out


class Preprocessed(RAW_Edge):
    def __init__(self, root, image_list_file, image_list):
        super(Preprocessed, self).__init__(root, image_list_file, image_list)

    def pack_raw(self, raw):
        # pack Bayer image to 4 channels
        def invDemosaic(img):
            img_R = img[::2, ::2, 0]
            img_G1 = img[::2, 1::2, 1]
            img_G2 = img[1::2, ::2, 1]
            img_B = img[1::2, 1::2, 2]
            raw_img = np.ones(img.shape[:2])
            raw_img[::2, ::2] = img_R
            raw_img[::2, 1::2] = img_G1
            raw_img[1::2, ::2] = img_G2
            raw_img[1::2, 1::2] = img_B
            return raw_img

        raw = 255. - raw
        raw /= 255.
        raw = raw[:, :, ::-1]
        im = invDemosaic(raw)

        if np.average(im) < 0.1:
            im *= (0.1 / np.average(im)) # scale image

        im = np.expand_dims(im, axis=2)
        img_shape = im.shape
        H = img_shape[0]
        W = img_shape[1]

        out = np.concatenate((im[0:H:2, 0:W:2, :],
                              im[0:H:2, 1:W:2, :],
                              im[1:H:2, 1:W:2, :],
                              im[1:H:2, 0:W:2, :]), axis=2)
        return out
