# coding:utf-8
import numpy as np
from yolov3.yolo_detector import face_detect as yolodetector
from PIL import Image
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from yolov3.utils import rescale_boxes
import torch
import cv2
import torchvision.transforms as transforms
import dlib

class FaceDetection(object):
    def __init__(self):
        super(FaceDetection, self).__init__()
        self.yolodetecter = yolodetector
        self.image_size = 150
        self.classes = ['face_mask']
        self.predictor = dlib.shape_predictor(r'./shape_predictor_68_face_landmarks.dat')

    def point_draw(self, img, sp, title, save): # Draw 68 key points in the image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range(68):
            cv2.putText(img, str(i), (sp.part(i).x, sp.part(i).y), cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 0, 255), 1,
                        cv2.LINE_AA)
        if save:
            filename = title+'.jpg'
            print('filename{}'.format(filename))
            cv2.imwrite(filename, img)
            cv2.imshow(title, img)
            cv2.waitKey(1000)

    def getfacefeature(self, image):  # Detect the face position and return the coordinate information
        print("\nPerforming object detection:")
        prev_time = time.time()  # Time to start loading the image.
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = transforms.ToTensor()(rgb_img)

        # face detection
        face_bboxes = self.yolodetecter(img)

        current_time = time.time()
        # timedelta represents the time difference between two datetimes
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        print("\t+ Inference Time: %s" % (inference_time))

        if face_bboxes[0] is None:
            cv2.putText(image, 'no face:', (100, 100), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            if len(face_bboxes[0])==1:
                detections = rescale_boxes(face_bboxes[0], 416, img.shape[1:])
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    print(x1, y1, x2, y2)
                    face_bbox = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))
                    print('faces{}'.format(face_bbox))
                    cv2.rectangle(rgb_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

                    # key points
                    shape_68 = self.predictor(rgb_img, face_bbox)
                    self.point_draw(rgb_img, shape_68, 'before_image_warping', save=True)
                    print("Computing descriptor on aligned image ..")
                    # face alignment
                    images_align = dlib.get_face_chip(rgb_img, shape_68, size=150)
                    images_align = np.array(images_align).astype(np.uint8)
                    img_align = transforms.ToTensor()(images_align)
                    face_bboxes = self.yolodetecter(img_align)
                    if len(face_bboxes[0]) == 1:
                        detections = rescale_boxes(face_bboxes[0], 416, img_align.shape[1:])
                        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                            face_bbox = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))
                        point68_dlib = self.predictor(images_align, face_bbox)
                        self.point_draw(images_align, point68_dlib, 'after_image_warping', save=True)

        return face_bboxes


if __name__=='__main__':
    '''******* image ******'''
    # path = 'D:\Python\project3\master_project\\face_detection\\face-detect\\1_0_1.jpg'
    # image = cv2.imread(path)
    # facedetection = FaceDetection()
    # face_bboxes = facedetection.getfacefeature(image)
    # print(face_bboxes)



    '''******* Take an image from the video ****************************'''
    cap = cv2.VideoCapture(0)
    print('cap.isOpened(){}'.format(cap.isOpened()))
    while (cap.isOpened()):
        # get a frame
        ret, frame = cap.read()
        # show a frame
        if ret == True:
            cv2.imshow('real_person', frame)
            # select one frame
            if cv2.waitKey(100) & 0xFF == ord('s'):
                print('select one image successful!')
                image = frame
                facedetection = FaceDetection()
                face_bboxes = facedetection.getfacefeature(image)
                print('face_position:{}'.format(face_bboxes[0]))

            if cv2.waitKey(10) & 0xFF == ord('q'):
                print('exit')
                break
        else:
            cap.release()
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

    '''********* video *************'''
    # cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("D:\Python\project3\master_project\\face_detection\\face-detect\\1.mp4")
    # print('cap.isOpened(){}'.format(cap.isOpened()))
    # while (cap.isOpened()):
    #     # get a frame
    #     ret, frame = cap.read()
    #     # show a frame
    #     if ret == True:
    #         facedetection = FaceDetection()
    #         face_bboxes = facedetection.getfacefeature(frame)
    #
    #         if cv2.waitKey(10) & 0xFF == ord('q'):
    #             print('exit')
    #             break
    #     else:
    #         cap.release()
    #         cv2.waitKey(0)
    #
    # cap.release()
    # cv2.destroyAllWindows()