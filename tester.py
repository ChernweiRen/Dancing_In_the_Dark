import os
import gc

import numpy as np
import torch
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
        # import pdb;pdb.set_trace()
        
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
                cv2.imwrite(
                    os.path.join(
                        self.result_dir,
                        f'{os.path.basename(img_file.replace("./", "").replace("/", "_"))[:-4]}_out.png'
                    ),
                    np.clip(out, 0, 255).astype(np.uint8)
                )

