# Author: Zylo117
"""
Simple Inference Script of EfficientDet-Pytorch
"""
import sys
sys.path.append('/workspace/Yet-Another-EfficientDet-Pytorch')

import os
import time
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, plot_one_box


def prepare_dirs(*dirs):
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)


def model_init(args):
    compound_coef = args.compound_coef
    checkpoint = args.checkpoint
    use_cuda = not args.cpu
    cudnn.fastest = True
    cudnn.benchmark = True

    # replace this part with your project's anchor config
    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

    # tf bilinear interpolation is different from any other's, just make do
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=90,
                                 ratios=anchor_ratios, scales=anchor_scales)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model.requires_grad_(False)
    model.eval()
    if use_cuda:
        model = model.cuda()

    return model


def detect(model, dataset, args):
    use_cuda = not args.cpu
    threshold = args.threshold
    iou_threshold = args.iou_threshold
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[args.compound_coef]

    img_dir = os.path.join(dataset, dataset, 'images')
    bbox_dir = os.path.join( dataset, dataset, 'annotations', 'bboxes')
    vis_dir = os.path.join( dataset, 'det_vis')
    prepare_dirs(bbox_dir, vis_dir)


    img_paths = [ os.path.join(img_dir, f) for f in os.listdir(img_dir) ]
    for img_path in tqdm(img_paths):
        ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)
        ori_img = ori_imgs[0]
        img_id = os.path.basename(img_path).split('.')[0]

        json_byhand = os.path.join(dataset, 'annotation_byhand', img_id+'.json')
        if os.path.exists(json_byhand):
            with open(json_byhand) as f:
                annotation_byhand = json.load(f)
                points = annotation_byhand['shapes'][0]['points']
                max_box = points[0] + points[1]
        else:
            if args.update:         # only process annotations by hand
                continue
            if use_cuda:
                x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
            else:
                x = torch.stack([torch.from_numpy(ft) for fi in framed_imgs], 0)

            x = x.to(torch.float32).permute(0, 3, 1, 2)

            with torch.no_grad():
                features, regression, classification, anchors = model(x)

                regressBoxes = BBoxTransform()
                clipBoxes = ClipBoxes()

                preds = postprocess(x,
                                anchors, regression, classification,
                                regressBoxes, clipBoxes,
                                threshold, iou_threshold)

                pred = invert_affine(framed_metas, preds)[0]

            max_area, max_box = 0, [ 0, 0, ori_img.shape[1], ori_img.shape[0] ]
            for det, class_id in zip(pred['rois'], pred['class_ids']):
                if not class_id == 0:
                    continue
                x1, y1, x2, y2 = det.astype(np.int)
                w, h = x2-x1, y2-y1
                area = w*h
                if area > max_area:
                    max_area = area
                    max_box = [x1, y1, x2, y2]

        
        plot_one_box(ori_img, max_box, color=[255,0,255], line_thickness=2)
        if args.vis:
            cv2.imwrite( os.path.join(vis_dir, img_id+'.jpg'), ori_img )

        bbox_file = os.path.join(bbox_dir, img_id+'.txt')
        with open( bbox_file, 'w' ) as f:
            bbox_info = ' '.join( map(str, max_box) )
            f.write(bbox_info)


def main():
    parser = argparse.ArgumentParser()
    # datasets
    parser.add_argument('--datasets', nargs='+')
    # model init
    parser.add_argument('--compound_coef', type=int, default=8, help='model selection: 0-8')
    parser.add_argument('--checkpoint', type=str, default='/workspace/Yet-Another-EfficientDet-Pytorch/weights/efficientdet-d8.pth')
    parser.add_argument('--cpu', action='store_true')
    # model infer
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--iou_threshold', type=float, default=0.2)
    parser.add_argument('--update', action='store_true', help='only process annotations by hand')
    # other
    parser.add_argument('--vis', action='store_true', default=False)
    
    args = parser.parse_args()
    datasets = args.datasets

    model = model_init(args)
    for dataset in datasets:
        detect(model, dataset, args)


if __name__ == "__main__":
    main()