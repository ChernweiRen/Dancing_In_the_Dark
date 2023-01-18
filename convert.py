import argparse
import os
import cv2
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Low light edge extraction")
    parser.add_argument('--path', type=str, default='.', help='pics path')
    args = parser.parse_args()

    path = args.path

    if not os.path.exists(os.path.join(path, 'npy')):
        os.mkdir(os.path.join(path, 'npy'))
    
    # print(os.listdir(path))
    for each_pic in os.listdir(path):
        print(each_pic)
        try:
            img = cv2.imread(os.path.join(path, each_pic), 0)
            img = img.astype(np.uint16)
            img *= 4
            # import pdb;pdb.set_trace()
            np.save(os.path.join(path, 'npy', '%s.npy'%each_pic[:-4]), img)
        except:
            pass
        
