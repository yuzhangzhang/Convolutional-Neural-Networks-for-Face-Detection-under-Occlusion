# coding:utf-8
# 路径置顶
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.getcwd())
from torch.nn.modules.distance import PairwiseDistance
import numpy as np
import time
import cv2
import torchvision.transforms as transforms
import torch
import dlib
import argparse
from facenet.face_feature_extractor import FaceFeatureExtractor
from yolov3.yolo_detector import face_detect as yolodetector
from yolov3.utils import rescale_boxes


class FaceRecognition(object):
    def __init__(self):
        super(FaceRecognition, self).__init__()
        self.yolodetecter = yolodetector
        self.image_size = 256
        self.classes = ['face_mask']
        self.predictor = dlib.shape_predictor(r'./shape_predictor_68_face_landmarks.dat')
        self.face_feature_extractor = FaceFeatureExtractor


    def FacePositionDetect(self,image):
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = transforms.ToTensor()(rgb_img)
        # face detection
        face_bboxes = self.yolodetecter(img)
        if face_bboxes[0] is None:
            cv2.putText(image, 'no face:', (100, 100), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            if len(face_bboxes[0])==1:
                detections = rescale_boxes(face_bboxes[0], 416, img.shape[1:])
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    print(x1, y1, x2, y2)
                    face_bbox = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))
                    print('face_bbox:{}'.format(face_bbox))
                    # cv2.rectangle(rgb_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                    # key point extraction
                    shape_68 = self.predictor(rgb_img, face_bbox)
                    print("Computing descriptor on aligned image ..")
                    # face alignment
                    images_align = dlib.get_face_chip(rgb_img, shape_68, size=self.image_size)
                    images_align = np.array(images_align).astype(np.uint8)
                    images_align = cv2.cvtColor(images_align, cv2.COLOR_BGR2RGB)
                    cv2.imshow('align', images_align)
                    cv2.waitKey(10000)
                    return images_align

    def load_personnel_face_information(self,personnel_face_embeddings_path):
        # 读入已记录的人员数据库  Read into the recorded personnel information
        personnel_face_embeddings = []
        personnel_face_images = []
        if os.path.exists(personnel_face_embeddings_path):
            fi = np.load(personnel_face_embeddings_path,allow_pickle=True)
            for w in fi:
                embeddings, images = w
                personnel_face_embeddings.append(embeddings)
                personnel_face_images.append(images)
        print(len(personnel_face_embeddings))
        return personnel_face_embeddings, personnel_face_images

    def MakePersonnelFaceEmbeddings(self,database_path,personnel_face_embeddings_path,new_personnel_face_images_path):
        personnel_face_embeddings=[]
        personnel_face_images=[]
        files = os.listdir(database_path)  # 得到文件夹下的所有文件名称 Get the names of all files in the folder
        with torch.no_grad():
            for file in files:
                image_path = database_path+'\\'+file
                image = cv2.imread(image_path)
                # face feature extraction
                align_img = self.FacePositionDetect(image)
                path = new_personnel_face_images_path+'\\'+file
                # print(path)
                cv2.imwrite(path,align_img)
                face_embedding = self.face_feature_extractor(align_img)
                face_embedding = face_embedding.numpy()
                personnel_face_embeddings.append(face_embedding)
                personnel_face_images.append(image_path)
            # updating the personnel information file
            print("updating npy file...")
            combine = []
            for embedding, images in zip(personnel_face_embeddings, personnel_face_images):
                combine.append(
                    [
                        embedding,
                        images
                    ]
                )
            np.save(personnel_face_embeddings_path, combine)
            print("npy file Saved!\n")
        return personnel_face_embeddings,personnel_face_images

    # Add personnel embedding
    def AddNewFaceFeature(self, face_image, personnel_face_images, personnel_face_embeddings, personnel_face_images_path, personnel_face_embeddings_path):
        with torch.no_grad():  # 不传梯度了
            print('please input the name:')
            while True:
                line = input()
                if line == ' ': break
                name = line
                path = personnel_face_images_path + '\\' + name + '.jpg'
                # print('path:{}'.format(path))
                # face_image=cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                # cv2.imshow('img',face_image)
                # cv2.waitKey(0)
                cv2.imwrite(path, face_image)
                personnel_face_images.append(path)

            # face feature extraction
            face_embedding = self.face_feature_extractor(face_image)
            face_embedding = face_embedding.numpy()
            personnel_face_embeddings.append(face_embedding)

            # updating the personnel information file
            print("updating npy file...")
            combine = []
            for embedding, images in zip(personnel_face_embeddings, personnel_face_images):
                combine.append(
                    [
                        embedding,
                        images
                    ]
                )
            np.save(personnel_face_embeddings_path, combine)
            print("npy file Saved!\n")

            return personnel_face_embeddings, personnel_face_images

    #
    def FaceFeatureComparision(self,face_image, personnel_face_embedding):
        l2_distance = PairwiseDistance(2)
        with torch.no_grad():
            Similarity = 0
            image_num = 0
            face_embedding = self.face_feature_extractor(face_image)
            dis = []
            sim = []
            for idx, embed in enumerate(personnel_face_embedding):
                face_embedding = torch.div(face_embedding, torch.norm(face_embedding))
                embed = torch.from_numpy(embed)
                embed = torch.div(embed, torch.norm(embed))

                distance = l2_distance.forward(face_embedding, embed)
                dis.append(distance)
                similarity = 1 - distance
                print('similarity:{}'.format(similarity))
                sim.append(similarity)
                if similarity > Similarity:
                    Similarity = similarity
                    image_num = idx
            # print('min_dis:{}'.format(np.min(dis)))
            # print('distance:{}'.format(dis))
            # _range = np.max(dis) - np.min(dis)
            # nor_dis = (dis - np.min(dis)) / _range
            # print('nor_dis:{}'.format(nor_dis))
            # sim = 1 - nor_dis
            # print('sim:{}'.format(sim))
            # image_num=np.argmin(nor_dis, axis=0)
            # print('img_num:{}'.format(image_num))
            # Similarity = 1- np.min(nor_dis)
            # print('Similarity:{}'.format(Similarity))
            return Similarity,image_num,sim


def ChooseMethod(method, image):
    facerecognition = FaceRecognition()
    if method == 'AddFace':
        # for image in images:
        images_align = facerecognition.FacePositionDetect(image)
        face_embeddings, face_images = facerecognition.load_personnel_face_information(opt.personnel_face_embeddings_path)
        face_embeddings, face_images = facerecognition.AddNewFaceFeature(images_align, face_images, face_embeddings,
                                                                         opt.new_personnel_face_images_path, opt.personnel_face_embeddings_path)
        return (face_embeddings, face_images)

    elif method == 'Identify':
        # img = cv2.imwrite('new.jpg',image)
        # image = cv2.imread('new.jpg')
        images_align = facerecognition.FacePositionDetect(image)
        face_embeddings, face_images = facerecognition.load_personnel_face_information(
            opt.personnel_face_embeddings_path)
        # print('embeddings:{},images:{}'.format(face_embeddings, face_images))
        Similarity, image_num, sim = facerecognition.FaceFeatureComparision(images_align, face_embeddings)
        # if Similarity>0.9:
        #     Similar_img = cv2.imread(face_images[image_num])
        #     cv2.putText(Similar_img, "simlarity:%1.5f" % (Similarity), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        #                 (0, 0, 255), 1, cv2.LINE_AA)
        #     cv2.imshow("similar_img", Similar_img)
        #     cv2.imshow("img", image)
        #     cv2.waitKey(10000)
        # else:
        #     cv2.putText(image, "There are no same people", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        #                 (0, 0, 255), 1, cv2.LINE_AA)
        #     cv2.imshow("img", image)
        #     cv2.waitKey(10000)
        #
        face_images = ['D:\Python\project3\master_project\\face_detection\\face-detect\\new_personnel_face_images\\test_00000002.jpg',
                       'D:\Python\project3\master_project\\face_detection\\face-detect\\new_personnel_face_images\\test_00000009.jpg',
                       'D:\Python\project3\master_project\\face_detection\\face-detect\\new_personnel_face_images\\test_00000018.jpg',
                       'D:\Python\project3\master_project\\face_detection\\face-detect\\new_personnel_face_images\\test_00000025.jpg',
                       'D:\Python\project3\master_project\\face_detection\\face-detect\\new_personnel_face_images\\test_00000038.jpg',
                       'D:\Python\project3\master_project\\face_detection\\face-detect\\new_personnel_face_images\\test_00000084.jpg',
                       'D:\Python\project3\master_project\\face_detection\\face-detect\\new_personnel_face_images\\zhangyu.jpg',
                       'D:\Python\project3\master_project\\face_detection\\face-detect\\new_personnel_face_images\\zhangyu2.jpg']
        Similar_img0 = cv2.imread(face_images[0])
        cv2.putText(Similar_img0, "simlarity:%1.5f" % (sim[0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("similar_img0", Similar_img0)

        Similar_img1 = cv2.imread(face_images[1])
        cv2.putText(Similar_img1, "simlarity:%1.5f" % (sim[1]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("similar_img1", Similar_img1)

        Similar_img2 = cv2.imread(face_images[2])
        cv2.putText(Similar_img2, "simlarity:%1.5f" % (sim[2]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("similar_img2", Similar_img2)

        Similar_img3 = cv2.imread(face_images[3])
        cv2.putText(Similar_img3, "simlarity:%1.5f" % (sim[3]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("similar_img3", Similar_img3)

        Similar_img4 = cv2.imread(face_images[4])
        cv2.putText(Similar_img4, "simlarity:%1.5f" % (sim[4]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("similar_img4", Similar_img4)

        Similar_img5 = cv2.imread(face_images[5])
        cv2.putText(Similar_img5, "simlarity:%1.5f" % (sim[5]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("similar_img5", Similar_img5)

        Similar_img6 = cv2.imread(face_images[6])
        cv2.putText(Similar_img6, "simlarity:%1.5f" % (sim[6]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("similar_img6", Similar_img6)

        cv2.waitKey(0)

        return Similarity
    else:
        print('Error! Please choose method again!')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--personnel_face_images_path", type=str, default="D:\Python\project3\master_project\\face_detection\\face-detect\personnel_face_images", help="path to face images file")
    parser.add_argument("--personnel_face_embeddings_path", type=str, default="personnel_face_embeddings_cbam.npy", help="path to face embeddings file")
    parser.add_argument("--personnel_database_path", type=str, default="D:\Python\project3\master_project\\face_detection\\face-detect\\test_img", help="path to face embeddings file")
    parser.add_argument("--new_personnel_face_images_path", type=str, default="D:\Python\project3\master_project\\face_detection\\face-detect\\new_personnel_face_images", help="path to face embeddings file")
    parser.add_argument("--method", type=str, default="Identify", help="AddFace,Identify")
    opt = parser.parse_args()
    print(opt)

    while True:
        print('Please choose the method: Identify, AddFace, DeletFace ?')
        while True:
            line = input()
            if line == ' ':
                break
            method = line
        if method =='DeletFace':
            print('Please input the name of delet image')
            line = input()
            name = line
            facerecognition = FaceRecognition()
            face_embeddings, face_images = facerecognition.load_personnel_face_information(opt.personnel_face_embeddings_path)
            personnel_face_images=[]
            personnel_face_embeddings=[]
            for path, embed in zip(face_images,face_embeddings):
                image_name = path.split('\\')[-1]
                image_name = image_name.split('.')[0]
                if image_name!=name:
                    print(image_name,name)
                    personnel_face_images.append(path)
                    personnel_face_embeddings.append(embed)
                else:
                    remove_path = opt.new_personnel_face_images_path+'\\'+image_name+'.jpg'
                    print(remove_path)
                    os.remove(remove_path)
    
            print("updating npy file...")
            combine = []
            for embedding, images in zip(personnel_face_embeddings, personnel_face_images):
                combine.append(
                    [
                        embedding,
                        images
                    ]
                )
            np.save(opt.personnel_face_embeddings_path, combine)
            print("npy file Saved!\n")
    
        else:
            cap = cv2.VideoCapture(0)
            print('cap.isOpened(){}'.format(cap.isOpened()))
            while (cap.isOpened()):
                # get a frame
                ret, frame = cap.read()
                cv2.imshow('real_person', frame)
                # show a frame
                if ret == True:
                    if cv2.waitKey(100) & 0xFF == ord('s'):
                        print('select one image successful!')
                        image = frame
                        output = ChooseMethod(method, image)
                        break
                else:
                    cap.release()
                    cv2.waitKey(0)
    
            cap.release()
            cv2.destroyAllWindows()




    # using one image to test
    # path = 'D:\Python\project3\master_project\\face_detection\\face-detect\\test_img\zhangyu2.jpg'
#     path = 'D:\Python\project3\master_project\MAFA\\test-images\images\\test_00000003.jpg'
#     image = cv2.imread(path)
#     # print(image)
#     output = ChooseMethod(opt.method, image)


    # import os
    # path = "D:\Python\project3\master_project\\face_detection\\face-detect\\test_img"
    # files = os.listdir(path)
    # for file in files[2:3]:
    #     # print(file)
    #     image_path = path+'\\'+file
    #     image = cv2.imread(image_path)
    #     # print(image)
    #     output = ChooseMethod(opt.method, image)


    # database_path = 'D:\Python\project3\master_project\\face_detection\\face-detect\\test_img'
    # new_personnel_face_embeddings_path = 'D:\Python\project3\master_project\\face_detection\\face-detect\\new_personnel_face_images'
    # facerecognition = FaceRecognition()
    # embeddings,paths = facerecognition.MakePersonnelFaceEmbeddings(database_path,opt.personnel_face_embeddings_path,new_personnel_face_embeddings_path)




