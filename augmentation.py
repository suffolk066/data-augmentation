import os
import glob
import click
import albumentations as A
import matplotlib.pyplot as plt
import cv2

from rich.traceback import install
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
install()
console = Console()
progress_bar = Progress(
    TextColumn('{task.description} [progress.percentage]{task.percentage:>3.0f}%'),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn('• [cyan]Remaining :'),
    TimeRemainingColumn(),
)

CHOICE_FORMAT_LIST = ['yolo']
CHOICE_TYPE_LIST = ['vflip', 'hflip', 'rotate', 'random']
DIR = 'data_input/'
SAVE_DIR = 'data_output/'
JPG = '.jpg'
TXT = '.txt'

class Augmentationer():
    def __init__(self, annotation_format:str, augmentation_type:str) -> None:
        self.ANNOTATION_FORMAT = annotation_format
        self.AUGMENTATION_TYPE = augmentation_type
        self.DICT_TYPE = {}
        for choice in CHOICE_TYPE_LIST:
            self.DICT_TYPE[choice.upper()] = choice.lower() + '_'
            
        self.aug_func = {
            'VFLIP': A.VerticalFlip(p=1),
            'HFLIP': A.HorizontalFlip(p=1),
            'ROTATE': A.RandomRotate90(p=1),
        }
    
    def get_data(self) -> dict:
        """data_input 폴더에서 이미지 파일과 텍스트 파일 불러오기"""
        data = {}
        image_list = glob.glob(f'{DIR}/*{JPG}')
        txt_list = glob.glob(f'{DIR}/*{TXT}')
        
        # 이미지와 텍스트 파일의 갯수가 일치하지 않으면 에러
        if len(image_list) != len(txt_list):
            raise ValueError("Mismatch between image and text file counts")
        
        for i, image in enumerate(image_list):
            data[image] = txt_list[i]

        return data

    def load_annotation(self, txt:str) -> str|list:
        """Annotation 파일 읽기"""
        with open(txt) as f:
            arr_string = f.readline().split()
            class_id = arr_string[0]
            annotaion = [[float(x) for x in arr_string[1:]] + [class_id]]
        return class_id, annotaion
    
    def set_aug_type(self, aug_type:str) -> A.Compose:
        """입력 타입에 따라 상하, 좌우 반전 or 90도 회전 변환"""
        return A.Compose([self.aug_func[aug_type.upper()]], bbox_params=A.BboxParams(format=self.ANNOTATION_FORMAT))
    
    def set_random_aug(self) -> A.Compose:
        """랜덤으로 Data Augmentation 수행"""
        transform = A.Compose([
            A.OneOf([A.HorizontalFlip(p=1),
                A.RandomRotate90(p=1),
                A.VerticalFlip(p=1)], 
                p=1),
            ],
            bbox_params=A.BboxParams(format=self.ANNOTATION_FORMAT))
        return transform
    
    def save(self, save_txt:str, save_img:str, bbox:str, image:dict) -> None:
        """저장 경로에 파일 저장"""
        with open(save_txt, 'w+') as f:
            f.write(bbox)
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_img, imgRGB)
        
    def run(self, aug_type:str, data:dict) -> None:
        """Data Augmentation 수행"""
        # 변환 타입 체크
        if aug_type.upper() not in self.DICT_TYPE:
            raise ValueError(f'aug_type {aug_type} is not defined. Use one of {list(self.DICT_TYPE.keys())}')
        
        # save_prefix와 변환 할 transform 타입 설정
        save_prefix = self.DICT_TYPE[aug_type.upper()]
        transform = self.set_aug_type(aug_type) if aug_type.lower() != 'random' else self.set_random_aug()
        
        with progress_bar as p:
            task = p.add_task('[cyan]Current Progress', total=len(data))
            for image, txt in data.items():                  
                # Annotation 불러오기
                class_id, bbox = self.load_annotation(txt)
                image = plt.imread(image)
                # h, w, _ = image.shape

                # 데이터 변환 적용
                transformed = transform(image=image, bboxes=bbox)
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']
                x1, y1, x2, y2, class_id = transformed_bboxes[0]
                new_bbox = f'{class_id} {x1:6f} {y1:6f} {x2:6f} {y2:6f}'
                
                # 시각화
                # plt.imshow(transformed_image)
                # plt.xticks([]); plt.yticks([])
                # plt.show()
                
                # 저장할 이름 설정
                original_file_name = os.path.splitext(os.path.basename(txt))[0]
                save_file_name = os.path.join(SAVE_DIR, save_prefix + original_file_name)
                save_txt = str(save_file_name + TXT)
                save_img = str(save_file_name + JPG)
                
                # 이미지 & 텍스트 저장
                self.save(save_txt, save_img, new_bbox, transformed_image)
                p.update(task, advance=1)
                
    def main(self) -> None:
        data = self.get_data()
        self.run(self.AUGMENTATION_TYPE, data)
        
@click.command()
def execute():
    f = click.prompt('Please enter a Annotation Format to Transform: ', default=CHOICE_FORMAT_LIST[0], type=click.Choice(CHOICE_FORMAT_LIST))
    t = click.prompt('Please enter a Augmentation Type to Transform: ', default=CHOICE_TYPE_LIST[3], type=click.Choice(CHOICE_TYPE_LIST))
    Augmentationer(f, t).main()
    
if __name__ == '__main__':
    execute()