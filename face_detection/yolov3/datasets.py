import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from yolov3.augmentations import horisontal_flip

# from GridMask import GridMask
# from Mosaic import load_mosaic
# from augmentations import horisontal_flip

from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):  # Turn the picture into a square
    c, h, w = img.shape
    dim_diff = np.abs(h - w)  # difference between h and w
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2 # Short one plus half of the difference, long one minus half of the difference
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Dimensions (left, right, top, bottom), if the width is greater than the height, add half of the difference between
    # the upper and lower sides, if the height is greater than the width, add half of the difference between the left and right sides.
    # Hout​=Hin​+padding_top+padding_bottom  Wout​=Win​+padding_left+padding_right

    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value) #vAlue is only valid when the mode is ‘constant’, which means that the filled value is constant and the value is 'value'
    # print('img_shape{}'.format(img.shape))
    return img, pad


def resize(image, size):  # Upsampling or downsampling, becomes size
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):  # Generate picture path and resized picture
    def __init__(self, img, img_size=416):
        # self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img = img
        self.img_size = img_size

    def __getitem__(self, index):
        # Pad to square resolution
        img, _ = pad_to_square(self.img, 0) # Turn the picture into a square and fill it with 0
        # If resize directly, will directly lose the image information. The network has to adapt to the problem of image
        # size in the process of learning classification, resulting in poor training effect. In yolo3, the height and width are adjusted to the same size.
        # In the resize of the up-sampling, the coordinate position of the label should be modified at the same time,
        # the horizontal flip is random, the size is randomly changed again, and then the size of 416 is changed as input.

        # Resize
        img = resize(img, self.img_size) # Turn the image into 416*416 size

        return img

    def __len__(self):
        return len(self.img)

class ListDataset(Dataset):  #Define the data set and labels required for training
    def __init__(self, list_path, img_size=416, data_augmentation=0,augment=True, multiscale=True, normalized_labels=True, maxepoch=1):
        with open(list_path, "r") as path:
            self.img_files = path.readlines()
        # n = len(self.img_files)
        # start = int(n/4)
        # end = int(start*3)
        # self.img_files = self.img_files[0:10]
        # print(self.img_files)

        self.label_files = [
            path.replace("train-images\images", "train-images\labels").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        # print(self.label_files)

        self.img_size = img_size
        self.max_objects = 100 # Define the maximum number of boxes contained in each picture
        self.data_augmentation = data_augmentation
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.data_augmentation = data_augmentation
        self.maxepoch = maxepoch



    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        # if img_path=='D:\Python\project3\master_project\MAFA\\train-images\images\\train_00011920.jpg' or img_path=='D:\Python\project3\master_project\MAFA\\train-images\images\\train_00011922.jpg':
        #     pass
        # print('img_path:{}'.format(img_path))
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        # print('img:{}'.format(img))
        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)#Directly understood as the width and height of the photo

        # Pad to square resolution
        img, pad = pad_to_square(img, 0) #This step is to make the height and width the same size, pad: (left, right, up, down)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()
        # label_path='D:\Python\project3\master_project\MAFA\\train-images\labels\\train_00011922.txt'
        # label_path=='D:\Python\project3\master_project\MAFA\\train-images\images\\train_00011922.jpg':
        # print(label_path)
        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # print(boxes)
            # Extract coordinates for unpadded + unscaled image
            # The coordinate point position of the label is x1, y1, x2, y2, so first transform
            # No normalization
            boxes[:,0] = 1
            x1 = boxes[:, 1]
            y1 = boxes[:, 2]
            x2 = boxes[:, 3]
            y2 = boxes[:, 4]
            box_w = x2-x1
            box_h = y2-y1
            # Adjust for added padding
            # The size of the image has changed, so the coordinate points of the frame need to be modified
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]

            # boxes[:, 0] => class id
            # Returns (xc, yc, w, h)
            # The size of the image has changed, so the coordinates of the boxes need to be modified and the bbox updated. Normalized
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w     # newer center x
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h        # newer center y
            boxes[:, 3] = box_w / padded_w              # newer width
            boxes[:, 4] = box_h / padded_h              # newer height
            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes # The last five columns are boxes. The updated coordinates are filled into the occupied space.
            # print('target:{}'.format(targets[:,1:]))
            '''
            tensor([[0.0000,1.0000,0.3200,0.6500,0.1400,0.0860],
                    [0.0000,1.0000,0.1860,0.6410,0.0480,0.4000],
                    [0.0000,1.0000,0.2170,0.6490,0.0700,0.4800]])
            '''
        # Apply augmentations
        if self.data_augmentation == 2 or self.data_augmentation == 3 :
            img, targets = load_mosaic(index,self.img_files,self.label_files)

        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets) #Random horizontal flip

        if self.data_augmentation == 1 or self.data_augmentation == 3 :
            grid = GridMask(d1=96, d2=224, rotate=360, ratio=0.6, mode=1, prob=0.8)
            epoch = np.random.randint(self.maxepoch)
            grid.set_prob(epoch, 240)
            img = grid(img)
        return img_path, img, targets

    def collate_fn(self, batch):   #Functions in custom classes are used for batch processing
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets): # get image ID
            boxes[:, 0] = i  # Tell each box which image they belong to from 0~batch_size
                             # Boxes are 1*6 in targets, and the first number stores the imageID
            '''
            tensor([[0.0000,1.0000,0.5120,0.5740,0.9720,0.4300],
                    [1.0000,1.0000,0.5730,0.6430,0.4940,0.1740],
                    [2.0000,0.0000,0.7070,0.5040,0.2480,0.7410],
                    [3.0000,1.0000,0.3200,0.6500,0.1400,0.0860],
                    [3.0000,1.0000,0.1860,0.6410,0.0480,0.4000],
                    [3.0000,1.0000,0.2170,0.6490,0.0700,0.4800]])
            The final output of targets is a list. Each element of the list is n targets corresponding to an image 
            (this is a tensor), and target[:,0]=0 (that is, the first target of the aforementioned targets is 0), 
            target[:,0] represents the ID of the corresponding image. During training, the collate_fn function will 
            merge all targets into a tensor (targets = torch.cat(targets, 0)), and only the first digit of this tensor 
            (target[:,0]) can judge this Which picture the target belongs to (that is, it can match the image ID)
            '''

        targets = torch.cat(targets, 0)  # Combine the target of each picture
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            # The collate_fn function is mainly to adjust the size of imgs, because YOLOv3 uses multi-scale training
            # during the training process, constantly changing the resolution of the image, making YOLOv3 suitable for image detection of various resolutions.
            # When training, use 32 pixels to zoom in and out
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)-1




if __name__=='__main__':
    train_path = 'D:\Python\project3\master_project\MAFA\\train.txt'
    dataset = ListDataset(train_path,img_size=416,augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )
    for i, (path,img,target) in enumerate(dataloader):
        print('************************************')
        print(i)
        # print(i)
        # print(target)
        # x,y = target.shape
        # for a in range(x):
        #     for b in range(2,y):
        #         if target[a,b]>1:
        #             print('**************error')
        #             print(target[a,b],a,b)

    print('finish')