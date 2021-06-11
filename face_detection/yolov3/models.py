from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from yolov3.parse_config import *
from yolov3.utils import build_targets, to_cpu, non_max_suppression
from yolov3.DropBlock import LinearScheduler,DropBlock2D
from yolov3.Mish_Swish import Swish,Mish

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_modules(module_defs,regularization,activations):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    # module_defs = [{"type":"net", "channels":3, ...},                     # each elemnt is a layer block (dtype=dict)
    #                {"type":"convolutional", "batch_normalize":1, ...},
    #                ...]

    hyperparams = module_defs.pop(0)                    # [net] overall parameters
    output_filters = [int(hyperparams["channels"])]     # 3: Initially, because it was rgb 3 channels
    module_list = nn.ModuleList()   # Store each layer, such as conv layer: including conv-bn-leaky relu, etc.
    # nn.ModuleList() & nn.Sequential()
    # nn.ModuleList(): It is the list of Module, which does not implement the forward function (there is no function that is actually executed),
    #                  so it is just the list of modules, and does not require the order relationship between modules
    # nn.Sequential(): The order of module execution. The forward function is implemented, that is, the modules in it will be executed sequentially,
    #                  so the size of each module must match

    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()   # Save the execution of each major layer, such as conv layer: including conv-bn-leaky relu, etc.
        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])    # Number of output channels
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}", # a newer formatting method for python3.x, called f-string. Better than %s..
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":     #Different activation functions can be selected
                if activations == 1 :
                    modules.add_module(f"leaky_{module_i}", Swish())
                elif activations == 2 :
                    modules.add_module(f"leaky_{module_i}", Mish())
                else:
                    modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))
            if regularization ==2 or regularization ==3:
                drop_prob = 0.1
                block_size = 3
                dropblock_layer = DropBlock2D(drop_prob, block_size)
                modules.add_module(f"drop_block_{module_i}", dropblock_layer)

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])  # Add the number of channels, corresponding to concat
            modules.add_module(f"route_{module_i}", EmptyLayer())

        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())

        elif module_def["type"] == "yolo":
            # # mask: 6,7,8 / 3,4,5 / 0,1,2 <=> small/middle/big feature map <=> big/middle/small object
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)] # get w & h
            anchors = [anchors[i] for i in anchor_idxs]     # len=3, 3 anchors per level, Based on 416
            # for mask: 6,7,8
            # [(116, 90), (156, 198), (373, 326)]
            num_classes = int(module_def["classes"])        # 80
            img_size = int(hyperparams["height"])           # 416
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size,regularization)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)         # Stored in each layer, such as conv corresponds to the execution of conv-bn-leaky relu,
        output_filters.append(filters)      # The output filter size of each layer is the number of channels. Initially it is 3, corresponding to rgb 3 channel

    return hyperparams, module_list


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x
    # There are at least two ways to upsample：
    # interpolate & transpose convolution
    # Interpolate has gradually become mainstream, because transpose convolution may produce chessboard-like artifacts (alias)


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416,regularization=0):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)     # 3
        self.num_classes = num_classes      # 80
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        # loss = a1*l_reg + a2*l_conf + a3*l_cls
        # l_conf = obj_scale*l_obj + noobj_scale * l_noobj
        self.obj_scale = 1                  # lambda
        self.noobj_scale = 100
        self.metrics = {}                   # A bunch of calculation variables
        self.img_dim = img_dim              # image size，416
        self.grid_size = 0  # grid size     # 13x13=>32, 26x26=>16, 52x52=>8
        self.regularization = regularization

    def compute_grid_offsets(self, grid_size, cuda=True):
        # 0<-13; 13<-26; 26<-52
        self.grid_size = grid_size
        g = self.grid_size          # 13, 26, 52
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size     # 32, 16, 8 => pixels per grid/feature point represents
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        # torch.arange(g): tensor([0,1,2,...,12])
        # torch.arange(g).repeat(g, 1):
        #       tensor([[0,1,2,...,12],
        #               [0,1,2,...,12],
        #               ...
        #               [0,1,2,...,12]])
        #       shape=torch.Size([13, 13])
        # torch.arange(g).repeat(g, 1).view([1, 1, g, g]):
        #       tensor([[[[0,1,2,...,12],
        #                 [0,1,2,...,12],
        #                 ...
        #                 [0,1,2,...,12]]]])
        #       shape=torch.Size([1, 1, 13, 13])
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        # torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]):
        #       tensor([[[[0,0,0,...,0],
        #                 [1,1,1,...,1],
        #                 ...
        #                 [12,12,12,...,12]]]])
        #       shape=torch.Size([1, 1, 13, 13])
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        # After FloatTensor(), the tuple() inside will become []
        # Change the anchor to the range (0, 13)
        # self.scaled_anchors = tensor([[3.625, 2.8125], [4.875, 6.1875], [11.65625, 10.1875]]) # 3x2
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        # self.scaled_anchors[:, :1]: tensor([[3.625], [4.8750], [11.6562]])
        # self.anchor_w =
        # self.scaled_anchors.view((1, 3, 1, 1)) =
        #                                          tensor([
        #                                                  [
        #                                                    [[3.625]],
        #                                                    [[4.8750]],
        #                                                    [[11.6562]]
        #                                                  ]
        #                                                 ])
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):
        # x.shape: b x 255 x 13 x 13 (anchor 6, 7, 8)

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)     # batch size
        grid_size = x.size(2)       # feature map size: 13, 26, 52  # initially, self.grid_size = 0
        # print(x.size())
        prediction = (
            #       b, 3, 85, 13, 13
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            #       b, 3, 13, 13, 85
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        # the x,y,w,h corresponds to the pink circle in slides (generated directly from network)
        x = torch.sigmoid(prediction[..., 0])  # Center x   # (b,3,13,13)            # 1 +
        y = torch.sigmoid(prediction[..., 1])  # Center y   # (b,3,13,13)            # 1 +
        w = prediction[..., 2]  # Width                     # (b,3,13,13)            # 1 +
        h = prediction[..., 3]  # Height                    # (b,3,13,13)            # 1 +
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf (b,3,13,13)            # 1 + = 5 +
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred. (b,3,13,13,80)    # 80 = 85

        # Initially, self.grid_size = 0 != 13, then 13 != 26, then 26 != 52
        # Each time, if former grid size does not match current one, we need to compute new offsets
        # effect：
        # 1. For feature maps of different sizes (13x13, 26x26, 52x52), find the coordinates of the upper left corner of different grids
        # 2. Scale the anchor in the range of (0, 416) to the range of (0, 13)
        #
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)
        # self.grid_x:                             # self.grid_y:
        #       tensor([[[[0,1,2,...,12],          #       tensor([[[[0,0,0,...,0],
        #                 [0,1,2,...,12],          #                 [1,1,1,...,1],
        #                 ...                      #                 ...
        #                 [0,1,2,...,12]]]])       #                 [12,12,12,...,12]]]])
        #       shape=torch.Size([1, 1, 13, 13])   #       shape=torch.Size([1, 1, 13, 13])
        #                                          #
        # self.anchor_w: shape([1, 3, 1, 1])       # self.anchor_h: shape([1, 3, 1, 1])
        # tensor([                                 # tensor([
        #         [                                #         [
        #           [[3.625]],                     #           [[2.8125]],
        #           [[4.8750]],                    #           [[6.1875]],
        #           [[11.6562]]                    #           [[10.1875]]
        #         ]                                #         ]
        #        ])                                #        ])

        # Add offset and scale with anchors
        # x, y, w, h are prediction, this part is directly derived from the network predict, and xy is forced to (0,1) through sigmoid
        # grid_xy is the coordinates of the upper left corner of the grid [0,1,...,12],
        # So xy+grid_xy is to distribute the pred result (ie the center point of the object) to each grid, (0, 13)
        #
        # For wh, since the result of prediction is directly after log(), exp is required here
        #
        # At this time, all pred_boxes are in the range (0,13)
        # These preds are final outpus for test/inference which corresponds to the blue circle in slides
        # This procedure could also be called as Decode
        #
        # Under normal circumstances, pure preds do not participate in the calculation of loss, but only exist as the final output.
        # But it is still calculated here and appears in the build_targets function. Its purpose is to assist in generating the mask
        pred_boxes = FloatTensor(prediction[..., :4].shape)     # (b, 3, 13, 13, 4)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (   # * stride (=32 for 13x13), the purpose is to restore the bbox of (0, 13) to (0, 416)
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0
        else:
            # iou_scores: [b, num_anchor, grid_size, grid_size] -> IoU of pred_boxes and ground_truth
            # class_mask: [b, num_anchor, grid_size, grid_size], The correct predicted class is true
            # obj_mask : [b, num_anchor, grid_size, grid_size] -> 1: Must be where the positive sample falls(b_id, anchor_id, i, j)
            #                                                  -> 0: It must not be where the positive sample falls
            # noobj_mask:  [b, num_anchor, grid_size, grid_size] -> 1: Must be where the negative sample falls
            #                                                    -> 0: Not necessarily where the positive sample falls, or it may not be involved in the calculation
            #                                                          Reflects the value of ignore_thres. >ignore, do not participate in the calculation
            # The following is the real target that is calculated to participate in generating loss (except tcls)
            # The procedure to generate those t·, corresponding to the gray circle in slides, can be called as Encode
            # tx: [b, num_anchor, grid_size, grid_size]
            # ty: [b, num_anchor, grid_size, grid_size]
            # tw: [b, num_anchor, grid_size, grid_size]
            # th: [b, num_anchor, grid_size, grid_size]
            # tcls :[b, num_anchor, grid_size, grid_size, n_classes]
            #

            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,              # (b, 3, 13, 13, 4)
                pred_cls=pred_cls,                  # (b, 3, 13, 13, 80)
                target=targets,                     # (n_boxes, 6) [details in build_targets function]
                anchors=self.scaled_anchors,        # (3, 2) 3 anchors, each with 2 dimensions
                ignore_thres=self.ignore_thres,     # 0.5 (hard code in YOLOLayer self.init())
                regularization = self.regularization,
            )
            obj_mask = obj_mask.bool()
            # print('obj:{}'.format(type(obj_mask)))
            noobj_mask = noobj_mask.bool()
            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            # Reg Loss
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])

            # Conf Loss
            # Because here conf chooses bce_loss, because noobj is basically predictable, so loss_conf_noobj is usually relatively small
            # So in order to balance at this time, noobj_scale is often greater than obj_scale, (100, 1)
            # In fact, the conf loss here is a 0-1 classification, 0 is noobj, 1 is obj
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj

            # Class Loss
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])

            # Total Loss
            # total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf

            print('total_loss:{},cls_loss:{},conf_loss:{},{},{},bbox:{},{},{},{}'.format(total_loss,loss_cls,loss_conf,loss_conf_obj,loss_conf_noobj,loss_x,loss_y,loss_h,loss_w))
            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()     # class_mask/obj_mask(b, 3, 13, 13) # accuracy
            conf_obj = pred_conf[obj_mask].mean()           # Average confidence of objects
            conf_noobj = pred_conf[noobj_mask].mean()       # Average confidence of no object
            conf50 = (pred_conf > 0.5).float()              # Positions with confidence greater than 0.5 (b, num_anchor, 13, 13)
            iou50 = (iou_scores > 0.5).float()              # The position where iou is greater than 0.5 (b, num_anchor, 13, 13)
            iou75 = (iou_scores > 0.75).float()             # The position where iou is greater than 0.75 (b, num_anchor, 13, 13)
            detected_mask = conf50 * class_mask * tconf     # tconf=obj_mask, That is: both the prediction confidence is> 0.5, and the class is also correct, and it is also obj
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, regularization,activations,img_size=416):
        super(Darknet, self).__init__()
        # Each element of the module_defs is a dict, a layer block with key values like 'type', 'batch_normalize', etc
        # module_defs = [{"type":"net", "channels":3, ...},                     # each elemnt is a layer block (dtype=dict)
        #                {"type":"convolutional", "batch_normalize":1, ...},
        #                ...]
        self.module_defs = parse_model_config(config_path)      # read in cfg where net is defined
        # hyperparams: {"type":"net", "channels":3, ...}
        # module_list: The sequential execution of each layer-block (excluding module_defs[0] (that is, [net]layer, that layer is hyperparams))
        # In create_modules, in order to extract hyperparams, hyper has been popped out, so module_defs has no [net]module at this time
        self.hyperparams, self.module_list = create_modules(self.module_defs,regularization,activations)
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")] # not used
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):
        # x = b*3*416*416
        img_dim = x.shape[2]        # 416
        loss = 0
        layer_outputs, yolo_outputs = [], []        # At this time module_defs has no [0] (net layer), it starts from conv
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                #              layer_outputs are the output of each module block
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]  # element-wise addition
            elif module_def["type"] == "yolo":
                # module[0] here: YOLOLayer.forward
                # Because module_list here corresponds .add_module(..., YOLOLayer), and it's under nn.Sequential,
                # so we need excute the .forward function
                x, layer_loss = module[0](x, targets, img_dim)      # targets: ground truth, from dataloader
                # x is predicted outputs
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()
