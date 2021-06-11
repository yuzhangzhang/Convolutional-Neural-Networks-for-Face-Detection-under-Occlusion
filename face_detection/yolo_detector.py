from __future__ import division
from yolov3.models import *
from yolov3.utils import *
from yolov3.datasets import *
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable



def face_detect(img):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="yolov3/yolov3-change.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="yolov3/yolov3_ckpt_mish.pth", help="path to weights file")
    parser.add_argument("--conf_thres", type=float, default=0.2, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.2, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument("--data_augmentation", default=0, help="0:None;1:GridMask;2:Mosaic;3:both of them")
    parser.add_argument("--regularization", default=0, help="0:None;1:labelsmoothing;2:dropblock;3:both of them")
    parser.add_argument("--activations", default=2, help="0:leaky_relu;1:swish;2:mish")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def,opt.regularization,opt.activations).to(device)

    model.load_state_dict(torch.load(opt.weights_path, map_location=lambda storage, loc: storage))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(img, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    print("\nPerforming object detection:")

    for batch_i, (input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))
        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
            if detections==None:
                print('no face')
            return detections
