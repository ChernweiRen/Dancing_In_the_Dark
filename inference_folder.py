#!/usr/bin/env python

import argparse
import gc
import numpy as np
import os

from numpy import imag
import torch
import torch.nn as nn

import datasets
import models.lsid as LSID
import models.dexined as DexiNed

from torch.autograd import Variable

import tqdm
import cv2

class Tester:
    def __init__(self, cuda, model, test_loader, result_dir=None):
        self.cuda = cuda
        self.model = model

        self.result_dir = result_dir
        self.test_loader = test_loader

    def test(self):
        self.model.eval()

        for (raws, img_files) in tqdm.tqdm(self.test_loader, total=len(self.test_loader), ncols=80, leave=False):
            gc.collect()
            if self.cuda:
                raws = raws.cuda()

            with torch.no_grad():
                raws = Variable(raws)
                output = self.model(raws)

            os.makedirs(self.result_dir, exist_ok=True)     
            
            for out, img_file in zip(torch.sigmoid(output), img_files):
                out = out.cpu().numpy().transpose(1, 2, 0) * 255
                # import pdb;pdb.set_trace()
                # print('result img save path:', os.path.join(self.result_dir,img_file.split('/')[-1:][0][:-4]+'_out.png'))
                cv2.imwrite(
                    os.path.join(
                        self.result_dir,
                        img_file.split('/')[-1:][0][:-4]+'_out.png'
                    ),
                    np.clip(out, 0, 255).astype(np.uint8)
                )


def get_parameters(model, bias=False):
    for k, m in model._modules.items():
        print("get_parameters", k, type(m), type(m).__name__, bias)
        if bias:
            if isinstance(m, nn.Conv2d):
                yield m.bias
        else:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                yield m.weight


def main():
    parser = argparse.ArgumentParser("Low light edge extraction")
    parser.add_argument('--arch_type', type=str, default='Sony', help='camera model type', choices=['Sony', 'Fuji', 'Canon', 'GC4653', 'Preprocessed'])
    parser.add_argument('--backbone', type=str, default='UNet', help='the backbone of the inference process', choices=['UNet', 'DexiNed'])
    parser.add_argument('--input_folder', type=str, default='', help='input raw file')
    parser.add_argument('--result_dir', type=str, default='./', help='directory where results are saved')
    parser.add_argument('--resume', type=str, default='', help='checkpoint file')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    args = parser.parse_args()

    resume = args.resume
    # print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()
    # if cuda:
    #     print("torch.backends.cudnn.version: {}".format(torch.backends.cudnn.version()))

    dataset_class = datasets.__dict__[args.arch_type]

    # if '/' in args.input:
    #     root = '/'.join(args.input.split('/')[:-1])
    #     test_img_list = args.input.split('/')[-1:]
    # else:
    #     root = './'
    #     test_img_list = [args.input]

    root = args.input_folder

    test_img_list = []
    for each_pic in os.listdir(root):
        test_img_list.append(os.path.join(root, each_pic))

    dt = dataset_class(root, image_list_file=None, image_list=test_img_list)
    test_loader = torch.utils.data.DataLoader(dt, batch_size=2, shuffle=False)

    if args.backbone == 'UNet':
        if 'Fuji' in args.arch_type: # Fuji
            model = LSID.lsid(inchannel=9, block_size=3)
        else: # Sony, Canon, GC4653
            model = LSID.lsid(inchannel=4, block_size=2)
        # print(model)
    elif args.backbone == 'DexiNed':
        if 'Fuji' in args.arch_type:
            bn = 1
            layer_num = 5
            fusion_layer = 6
            model = DexiNed.dexined(batch_norm=(bn), layer_num=layer_num, fusion_layer=fusion_layer,
                        inchannel=9, block_size=3)
        else: # Sony
            bn = 1
            layer_num = 5
            fusion_layer = 6
            model = DexiNed.dexined(batch_norm=(bn), layer_num=layer_num, fusion_layer=fusion_layer,
                        inchannel=4, block_size=2)
    print(model)

    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['model_state_dict'])
    checkpoint['arch'] = args.arch_type
    assert checkpoint['arch'] == args.arch_type
    # print("Resume from epoch: {}, iteration: {}".format(start_epoch, start_iteration))

    if cuda:
        model = model.cuda()

    # import hiddenlayer as h
    # vis_graph = h.build_graph(model, torch.zeros([1,4,1080,1920]).cuda())
    # vis_graph.theme = h.graph.THEMES["blue"].copy()
    # vis_graph.save('model_vis.png')
    # exit(0)
    

    tester = Tester(
        cuda=cuda,
        model=model,
        test_loader=test_loader,
        result_dir=args.result_dir
    )
    tester.test()
    print('Edge extraction complete!')

if __name__ == '__main__':
    main()
