import os
import glob
import click
import albumentations as A
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

class Augmentationer():
    def __init__(self, annotation_format:str, augmentation_type:str, is_show) -> None:
        self.ANNOTATION_FORMAT = annotation_format
        self.AUGMENTATION_TYPE = augmentation_type
        self.IS_SHOW = is_show
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
        image_list = glob.glob(f'{DIR}/*.jpg')
        txt_list = glob.glob(f'{DIR}/*.txt')
        
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
        return A.Compose([self.aug_func[aug_type.upper()]], \
                        bbox_params=A.BboxParams(format=self.ANNOTATION_FORMAT))
    
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

    def visualization(self, image:dict, bboxes:dict, dw:int, dh:int) -> None:
        plt.imshow(image)
        for bbox in bboxes:
            x_center, y_center, width, height, _ = bbox
            x1 = (x_center - width / 2) * dw
            y1 = (y_center - height / 2) * dh
            w = width * dw
            h = height * dh
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='red', fill=False)
            ax = plt.gca()
            ax.add_patch(rect)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def run(self, aug_type:str, data:dict) -> None:
        """Data Augmentation 수행"""
        # save_prefix와 변환 할 transform 타입 설정
        save_prefix = self.DICT_TYPE[aug_type.upper()]
        transform = self.set_aug_type(aug_type) if aug_type.lower() != 'random' else self.set_random_aug()
        
        with progress_bar as p:
            task = p.add_task('[cyan]Current Progress', total=len(data))
            for image, txt in data.items():
                # 저장할 이름 설정
                _image = os.path.basename(image)
                _text = os.path.basename(txt)
                _ext_img = os.path.splitext(_image)[1]
                _ext_txt = os.path.splitext(_text)[1]
                original_file_name = os.path.splitext(_text)[0]
                save_file_name = os.path.join(SAVE_DIR, save_prefix + original_file_name)
                save_img = str(save_file_name + _ext_img)
                save_txt = str(save_file_name + _ext_txt)
                   
                # Annotation 불러오기
                class_id, bbox = self.load_annotation(txt)
                image = plt.imread(image)

                # 데이터 변환 적용
                transformed = transform(image=image, bboxes=bbox)
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']
                h, w, _ = transformed_image.shape
                x1, y1, x2, y2, class_id = transformed_bboxes[0]
                new_bbox = f'{class_id} {x1:6f} {y1:6f} {x2:6f} {y2:6f}'
                
                # 시각화
                if self.IS_SHOW == True:
                    self.visualization(transformed_image, transformed_bboxes, w, h)
                
                # 이미지 & 텍스트 저장
                self.save(save_txt, save_img, new_bbox, transformed_image)
                p.update(task, advance=1)
                
        console.print('[bold green]Progress Done')
        
    def main(self) -> None:
        data = self.get_data()
        self.run(self.AUGMENTATION_TYPE, data)
        
@click.command()
def execute():
    click.clear()
    # default YOLO
    f = click.prompt('Please enter a Annotation Format: ', \
                    default=CHOICE_FORMAT_LIST[0], \
                    type=click.Choice(CHOICE_FORMAT_LIST))
    # default Random
    t = click.prompt('Please enter a Augmentation Type: ', \
                    default=CHOICE_TYPE_LIST[3], \
                    type=click.Choice(CHOICE_TYPE_LIST))
    c = click.confirm('Show visualization?')
    console.print('[bold magenta]Command :', f, t, c)
    Augmentationer(f, t, c).main()
    
if __name__ == '__main__':
    execute()