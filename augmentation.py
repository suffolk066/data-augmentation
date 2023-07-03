import albumentations as A
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
import cv2
import numpy as np

class Augmentationer():
    def __init__(self) -> None:
        self.DIR = 'data_input/'
        self.SAVE_DIR = 'data_output/'
        self.JPG = '.jpg'
        self.TXT = '.txt'
        self.DICT_FORMAT = {
            'VFLIP' : 'vflip_',
            'HFLIP' : 'hflip_',
            'ROTATE' : 'rotate_'
        }
    
    def get_data(self) -> dict:
        data = {}
        image_list = glob.glob(f'{self.DIR}/*.jpg')
        txt_list = glob.glob(f'{self.DIR}/*.txt')
        if len(image_list) != len(txt_list):
            return
        for i, image in enumerate(image_list):
            data[image] = txt_list[i]
        return data

    def load_annotation(self, txt:str) -> str|list:
        with open(txt) as f:
            arr_string = f.readline().split()
            class_id = arr_string[0]
            annotaion = [[float(x) for x in arr_string[1:]] + [class_id]]
        return class_id, annotaion
    
    def set_aug_type(self, aug_type) -> A.Compose:
        if aug_type == 'vflip':
            transform = A.Compose([
                A.VerticalFlip(p=1),
                ], 
                bbox_params=A.BboxParams(format='yolo'))
        elif aug_type == 'hflip':
            transform = A.Compose([
                A.HorizontalFlip(p=1),
                ], 
                bbox_params=A.BboxParams(format='yolo'))
        elif aug_type == 'rotate':
            transform = A.Compose([
                A.RandomRotate90(p=1),
                ], 
                bbox_params=A.BboxParams(format='yolo'))
        return transform
    
    def set_random_aug(self):
        transform = A.Compose([
            A.OneOf([A.HorizontalFlip(p=1),
                A.RandomRotate90(p=1),
                A.VerticalFlip(p=1)], 
                p=1),
            ],
            bbox_params=A.BboxParams(format='yolo'))
        return transform
    
    def save(self, save_txt:str, save_img:str, bbox:str, image:dict) -> None:
        with open(save_txt, 'w+') as f:
            f.write(bbox)
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_img, imgRGB)
        
    def run(self, data:dict) -> None:
        aug_type = 'vflip'
        for image, txt in data.items():
            # transform = self.set_aug_type(aug_type)
            transform = self.set_random_aug()
            if aug_type == 'vflip':
                save_prefix = self.DICT_FORMAT['VFLIP']
            elif aug_type == 'hflip':
                save_prefix = self.DICT_FORMAT['HFLIP']
            elif aug_type == 'rotate':
                save_prefix = self.DICT_FORMAT['ROTATE']

            original_file_name = str(txt).split('\\')[1].split('.')[0]
            save_file_name = str(self.SAVE_DIR + save_prefix + original_file_name)
            save_txt = str(save_file_name + self.TXT)
            save_img = str(save_file_name + self.JPG)
            
            # 데이터 로드
            class_id, bbox = self.load_annotation(txt)
            image = plt.imread(image)
            h, w, _ = image.shape

            # 변환 적용
            transformed = transform(image=image, bboxes=bbox)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            x1, y1, x2, y2, class_id = transformed_bboxes[0]
            new_bbox = f'{class_id} {x1:6f} {y1:6f} {x2:6f} {y2:6f}'
            
            # 시각화
            # plt.imshow(transformed_image)
            # plt.xticks([]); plt.yticks([])
            # plt.show()
            
            # 이미지 & 텍스트 저장
            self.save(save_txt, save_img, new_bbox, transformed_image)
    
    def main(self):
        data = self.get_data()
        self.run(data)

if __name__ == '__main__':
    aug = Augmentationer()
    aug.main()