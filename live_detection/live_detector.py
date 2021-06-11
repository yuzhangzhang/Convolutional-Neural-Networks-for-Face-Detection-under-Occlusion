# coding:utf-8
import dlib
import numpy as np
from copy import deepcopy
import cv2
import os
import torch
import sys
sys.path.append(r'E:\ex_python\occlusion\week3\code\week3\week3code-CVPR19-Face-Anti-spoofing')
from live_detection.model_baseline_SEFusion import FusionNet
import numpy as np
import torch.nn.functional as F
from imgaug import augmenters as iaa
from collections import OrderedDict
RESIZE_SIZE =112


def TTA_36_cropps(image, target_shape=(32, 32, 3)):
    image = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE))

    width, height, d = image.shape
    target_w, target_h, d = target_shape

    start_x = ( width - target_w) // 2   # 80
    start_y = ( height - target_h) // 2  # 80

    starts = [[start_x, start_y],      #(80,80)

              [start_x - target_w, start_y],   #(48, 80)
              [start_x, start_y - target_w],
              [start_x + target_w, start_y],
              [start_x, start_y + target_w],

              [start_x + target_w, start_y + target_w],
              [start_x - target_w, start_y - target_w],
              [start_x - target_w, start_y + target_w],
              [start_x + target_w, start_y - target_w],
              ]

    images = []

    for start_index in starts:
        image_ = image.copy()
        x, y = start_index

        if x < 0:
            x = 0
        if y < 0:
            y = 0

        if x + target_w >= RESIZE_SIZE:
            x = RESIZE_SIZE - target_w-1
        if y + target_h >= RESIZE_SIZE:
            y = RESIZE_SIZE - target_h-1

        zeros = image_[x:x + target_w, y: y+target_h, :]

        image_ = zeros.copy()

        zeros = np.fliplr(zeros)   #左右翻转
        image_flip_lr = zeros.copy()

        zeros = np.flipud(zeros)
        image_flip_lr_up = zeros.copy()

        zeros = np.fliplr(zeros)
        image_flip_up = zeros.copy()

        images.append(image_.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
        images.append(image_flip_lr.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
        images.append(image_flip_up.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
        images.append(image_flip_lr_up.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))

    return images

def color_augumentor(image, target_shape=(32, 32, 3), is_infer=False):
    if is_infer:
        augment_img = iaa.Sequential([
            iaa.Fliplr(0),
        ])

        image =  augment_img.augment_image(image)
        image = TTA_36_cropps(image, target_shape)
        return image


class FaceSpoofing():
    def __init__(self):
        self.net = FusionNet(num_class=2)
        model_path = 'D:\Python\project3\master_project\\face_detection\\face-detect -copy\live_detection\global_min_acer_model1.pth'
        # model_path='E:\ex_python\occlusion\week4\\new\global_min_acer_model.pth'
        # model_path='E:\ex_python\occlusion\week4\\new\Cycle_0_final_model.pth'
        if torch.cuda.is_available():
            state_dict = torch.load(model_path,map_location='cuda')
        else:
            state_dict = torch.load(model_path,map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        self.net.load_state_dict(new_state_dict)
        # self.net.load_state_dict(state_dict)
        if torch.cuda.is_available():
            self.net = self.net.cuda()
        self.net.eval()

    def classify(self,color):
        return self.detect(color)

    def detect(self,color):
        color = cv2.resize(color, (112, 112))
        color = color_augumentor(color, target_shape=(32, 32, 3), is_infer=True)
        n = len(color)
        image = np.concatenate(color, axis=0)
        image = np.transpose(image, (0, 3, 1, 2))
        image = image.astype(np.float32)
        image = image.reshape([n, 3, 32, 32])
        image = image / 255.0
        input_image = torch.FloatTensor(image)
        if (len(input_image.size()) == 4) and torch.cuda.is_available():
            input_image = input_image.unsqueeze(0).cuda()
        elif (len(input_image.size()) == 4) and not torch.cuda.is_available():
            input_image = input_image.unsqueeze(0)


        b, n, c, w, h = input_image.size()
        input_image = input_image.view(b * n, c, w, h)
        if torch.cuda.is_available():
            input_image = input_image.cuda()

        with torch.no_grad():
            logit, _, _ = self.net(input_image)
            logit = logit.view(b,n, 2)
            # print(logit)
            logit = torch.mean(logit, dim=1, keepdim=False)
            # print(logit)
            prob = 1-F.softmax(logit, 1)
            # value, top = prob.topk(1, dim=1, largest=True, sorted=True)

        print('probabilistic:',prob)
        print('predict: ', np.argmax(prob.detach().cpu().numpy()))
        return np.argmax(prob.detach().cpu().numpy())


def live_detect(image):
    face_spoofing = FaceSpoofing()
    live = face_spoofing.classify(image)
    # face_align = cv2.imread(face_align,cv2.COLOR_BGR2RGB)
    # if not face_spoofing.classify(image):
    #    print(" not humman\n ")
    #    # 框为红色
    #    cv2.putText(image,"Fake",(100,200),cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 1,cv2.LINE_AA)
    # else:
    #     cv2.putText(image, "Real", (100,200), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    # cv2.imshow('img',image)
    # cv2.waitKey(0)
    return live


def test():
    # 初始化人脸检测模型
    detector = dlib.get_frontal_face_detector()
    ## 填空 初始化活体检测模型
    face_spoofing = FaceSpoofing()
    # 初始化关键点检测模型
    predictor = dlib.shape_predictor(r'E:\ex_python\occlusion\week1\shape_predictor_68_face_landmarks.dat')
    # 初始化人脸特征模型
    recognition = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
    # 从摄像头读取图像, 若摄像头工作不正常，可使用：cv2.VideoCapture("week20_video3.mov"),从视频中读取图像
    cap = cv2.VideoCapture(0)
    while 1:
        # 初始化人脸相似度为-1
        similarity=-1
        # 读取图片
        ret, frame_src = cap.read()
        # 将图片缩小为原来大小的1/3
        x, y = frame_src.shape[0:2]
        # frame = cv2.resize(frame_src, (int(y / 3), int(x / 3)))
        frame = frame_src
        # face_align = frame_src
        # 使用检测模型对图片进行人脸检测
        dets = detector(frame, 1)
        #import pdb
        #pdb.set_trace()
        # 便利检测结果
        for det in dets:
            # 对检测到的人脸提取人脸关键点
            shape=predictor(frame, det)
            #print("x=%s,y=%s,w=%s,h=%s"%(det.left(),det.top(),det.width(),det.height()))
            # 在图片上绘制出检测模型得到的矩形框,框为绿色
            #import pdb
            #pdb.set_trace()
            # 人脸对齐
            face_align=dlib.get_face_chip(frame, shape, 150,0.1)
            ## 活体检测
            print('predict:{}'.format(face_spoofing.classify(face_align)))


            if not face_spoofing.classify(face_align):
               print(" not humman\n ")
               # 框为红色
               frame=cv2.rectangle(frame,(det.left(),det.top()),(det.right(),det.bottom()),(0,0,255),2)
               cv2.putText(frame,"Fake",(det.left(),det.top()),cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 1,cv2.LINE_AA)
            else:
                frame = cv2.rectangle(frame, (det.left(), det.top()), (det.right(), det.bottom()), (0, 255, 0), 2)
                cv2.putText(frame, "Real", (det.left(),det.top()), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)



        cv2.imshow("capture", frame)
        # cv2.imshow("face_align",face_align)
        if cv2.waitKey(10) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

