import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import argparse
import torch.nn as nn
import torchvision.transforms as transforms
# from vggface.model import resnet18_cbam
from vggface.model_CBAM import resnet18_cbam
# from vggface.model_fa_attention import resnet18_cbam

# from model import resnet18_cbam
import cv2
from PIL import Image
import numpy as np
import dlib

class DataProcess(Dataset):
    def __init__(self, img, transform=None):
        # 初始化
        self.img = img
        self.transform = transform
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('D:\Python\project3\master_project\\face_detection\\face-detect\shape_predictor_68_face_landmarks.dat')

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        # 如果做转换就转换一下下
        image = self.img
        # image = Image.fromarray(self.img).convert('RGB')  # cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # image = np.array(image).astype(np.uint8)
        # face_bbox = self.detector(image, 1)
        # if len(face_bbox) == 1:
        #     # 关键点提取
        #     shape_dlib = self.predictor(image, face_bbox[0])
        #     print("Computing descriptor on aligned image ..")
        #     # 人脸对齐 face alignment
        #     images_dlib = dlib.get_face_chip(image, shape_dlib, size=256)
        #     image = np.array(images_dlib).astype(np.uint8)
        #     cv2.imshow('img',image)
        #     cv2.waitKey(100)
        if self.transform:
            image = self.transform(image)
        return image

def FaceFeatureExtractor(img):
    parser = argparse.ArgumentParser()
    # parser.add_argument("--weights_path", type=str, default="D:\Python\project3\master_project\\face_detection\\face-detect\\vggface\model_resnet18_triplet_epoch_0_3.pt", help="path to weights file")
    parser.add_argument("--weights_path", type=str, default="D:\Python\project3\master_project\live_detection\\news\model_resnet18_triplet_mask_4.pt", help="path to weights file")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    # Set up model
    model = resnet18_cbam(pretrained=True, showlayer=False, num_classes=8631)
    input_features_fc_layer = model.fc.in_features
    model.fc = nn.Linear(input_features_fc_layer, 1280)

    if torch.cuda.is_available():
        model.cuda()
        print('Using single-gpu.')

    model_state = torch.load(opt.weights_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(model_state['model_state_dict'])
    print('loaded %s' % opt.weights_path)

    model.eval()  # Set in evaluation mode

    # 测试数据的变换
    test_data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    dataset = DataProcess(img, transform=test_data_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)

    print("\nPerforming object feature extraction:")

    for batch_index, face_img in enumerate(dataloader):
        face_embedding = model(face_img)
        return face_embedding


if __name__=='__main__':
    import cv2
    from torch.nn.modules.distance import PairwiseDistance

    img_path0 = 'D:\Python\project3\master_project\\face_detection\\face-detect\\3.jpg'
    img_path1 = 'D:\Python\project3\master_project\\face_detection\\face-detect\\2.jpg'

    # img_path1 = 'D:\Python\project3\master_project\\face_detection\\face-detect\\test_img\\test_00000002.jpg'
    img0 = cv2.imread(img_path0,cv2.COLOR_BGR2RGB)
    img1 = cv2.imread(img_path1,cv2.COLOR_BGR2RGB)

    embedding0 = FaceFeatureExtractor(img0)
    embedding1 = FaceFeatureExtractor(img1)

    l2_distance = PairwiseDistance(2)
    distance = l2_distance.forward(embedding0, embedding1)
    print('distance:{}'.format(distance))

    # from torch.nn.modules.distance import PairwiseDistance
    #
    # while True:
    #     cap = cv2.VideoCapture(0)
    #     print('cap.isOpened(){}'.format(cap.isOpened()))
    #     while (cap.isOpened()):
    #         # get a frame
    #         ret, frame = cap.read()
    #         cv2.imshow('real_person', frame)
    #         # show a frame
    #         if ret == True:
    #             if cv2.waitKey(100) & 0xFF == ord('s'):
    #                 print('select one image successful!')
    #                 image = frame
    #                 # cv2.imwrite('real_person.jpg', image)
    #                 embedding = FaceFeatureExtractor(image)
    #                 image1 = cv2.imread('D:\Python\project3\master_project\\face_detection\\face-detect\\vggface\\real_person.jpg')
    #                 embedding1 = FaceFeatureExtractor(image1)
    #                 l2_distance = PairwiseDistance(2)
    #                 distance = l2_distance.forward(embedding1, embedding)
    #                 print('distance:{}'.format(distance))
    #                 break
    #         else:
    #             cap.release()
    #             cv2.waitKey(0)
    #
    #     cap.release()
    #     cv2.destroyAllWindows()