#!/usr/bin/env python

import argparse
import os
import torch
import torch.nn as nn

import datasets
import models.lsid as LSID
import models.dexined as DexiNed
from tester import Tester


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
    parser.add_argument('--input', type=str, default='', help='input raw file')
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

    if '/' in args.input:
        root = '/'.join(args.input.split('/')[:-1])
        test_img_list = args.input.split('/')[-1:]
    else:
        root = './'
        test_img_list = [args.input]

    dt = dataset_class(root, image_list_file=None, image_list=test_img_list)
    test_loader = torch.utils.data.DataLoader(dt, batch_size=1, shuffle=False)

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
