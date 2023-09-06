from sklearn.model_selection import train_test_split
import yaml
import ultralytics
from ultralytics import YOLO
import os
from glob import glob
import shutil
from PIL import Image

# 코드와 같은 위치에 datasets 폴더, datasets 폴더 내부에는 6개의 데이터 폴더

PATH = 'datasets/'

convert_cls_num = {
    'snack': 0,
    'snack2': 20,
    'sauce': 28,
    'processed1':246,
    'processed2': 246,
    'myeon': 363,
    'can': 373
}

# 0~384까지로 클래스 재지정하여 딕셔너리 반환
def class_preprocess(PATH, folder_name):
    with open(PATH + folder_name + '/label/obj.names') as obj_names:
        classes = obj_names.readlines()
    obj_names.close()

    add_num = convert_cls_num[folder_name[9:]]

    obj_classes = {k + add_num: v.strip() for k, v in enumerate(classes)}

    

    for i in range(len(obj_classes)):
        if obj_classes[i + add_num] == '':
            del obj_classes[i + add_num]
    return obj_classes

# 종류별로 label 폴더 내 txt파일의 클래스 숫자 변경
def convert_label(PATH, folder_name):
    add_num = convert_cls_num[folder_name[9:]]
    label_list = glob(PATH + folder_name + '/label/*.txt')

    for label in label_list:
        if (label[-9:] != 'train.txt') and (label[-7:] != 'obj.txt'):
            print(label)
            with open(label) as f:
                lines = f.readlines()
            f.close()
    
            for i in range(len(lines)):
                line = lines[i]
                ary = line.split(' ')
                ary[0] = str(int(ary[0]) + add_num)
                lines[i] = ' '.join(ary)
            
            with open(label, 'w') as f:
                for line in lines:
                    f.write(line)
            f.close()

# YOLO-style 레이블 파일을 생성하는 함수
def convert_and_save_yolo_label(original_label_path, yolo_label_dir, image_width, image_height):
    with open(original_label_path, 'r') as original_label_file:
        lines = original_label_file.readlines()

    yolo_label = []
    for line in lines:
        class_id, x1, y1, x2, y2 = map(float, line.strip().split())
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        # 이미지 크기에 따라 좌표 스케일링
        x_center, y_center, width, height = x_center * image_width, y_center * image_height, width * image_width, height * image_height
        
        yolo_label.append(
            f'{int(class_id)} {x_center} {y_center} {width} {height}\n')

    yolo_label_filename = os.path.basename(
        original_label_path).replace('.txt', '.txt')
    yolo_label_path = os.path.join(yolo_label_dir, yolo_label_filename)

    with open(yolo_label_path, 'w') as yolo_label_file:
        yolo_label_file.writelines(yolo_label)


def to_yolov8(PATH, folder_name):
    """
    PATH = '....../datasets'
    사이킷런의 split 함수를 통해 데이터를 분류,
    datasets 안에 train, test, val 폴더 생성 후 모든 이미지와 라벨 복사.
    
    """

    
    img_list = glob(PATH + folder_name + '/label/*.jpg')
    label_list = glob(PATH + folder_name + '/label/*.txt')
    os.makedirs(PATH + '/train', exist_ok=True)
    os.makedirs(PATH + '/test', exist_ok=True)
    os.makedirs(PATH + '/val', exist_ok=True)
    yolo_label_dir = PATH + 'yolo_txt/'
    os.makedirs(yolo_label_dir, exist_ok=True)
    # 이미지 파일과 해당 레이블 파일에 대해 YOLO-style 레이블 생성 및 저장
    for original_label_path in glob(PATH + folder_name + '/label/*.txt'):
        if (original_label_path[-7:] != 'obj.txt') and (original_label_path[-9:] != 'train.txt'):
            image_path = original_label_path.replace('.txt', '.jpg')
            # 이미지 파일과 레이블 파일의 이름이 같다고 가정합니다.
            image_width, image_height = Image.open(image_path).size
            convert_and_save_yolo_label(original_label_path, yolo_label_dir, image_width, image_height)
            

    train_img_list, test_img_list = train_test_split(
        img_list, test_size=0.15, random_state=42)  # 0.85:0.15

    train_img_list, val_img_list = train_test_split(
        train_img_list, test_size=0.2, random_state=42)  # 8:2

    # train, val, test 폴더로 이미지와 라벨 이동
    for train_img in train_img_list:
        try:
            shutil.copy(train_img, PATH + '/train')
            train_txt = os.path.basename(train_img).replace('.jpg', '.txt')
            shutil.copy(train_img.replace('.jpg', '.txt'), PATH + '/train')
        except:
            pass

    for val_img in val_img_list:
        try:
            shutil.copy(val_img, PATH + '/val')
            val_txt = os.path.basename(val_img).replace('.jpg', '.txt')
            shutil.copy(val_img.replace('.jpg', '.txt'), PATH + '/val')
        except:
            pass

    for test_img in test_img_list:
        try:
            shutil.copy(test_img, PATH + '/test')
            test_txt = os.path.basename(test_img).replace('.jpg', '.txt')
            shutil.copy(test_img.replace('.jpg', '.txt'), PATH + '/test')
        except:
            pass
#    

# 각 카테고리별 클래스 정보 정리.        
obj_snack_1 = class_preprocess(PATH, 'dataset1_snack')
obj_snack_2 = class_preprocess(PATH, 'dataset2_snack2')
obj_sauce = class_preprocess(PATH, 'dataset3_sauce')
obj_processed = class_preprocess(PATH, 'dataset4_processed1')
#   processed2에는 obj.names가 없음. processed1과 통합인 것으로 예상
obj_myeon = class_preprocess(PATH, 'dataset6_myeon')
obj_can = class_preprocess(PATH, 'dataset7_can')

# 모든 카테고리에 대한 클래스 정보 합치기
obj_info = {**obj_snack_1, **obj_snack_2, **obj_sauce, **obj_processed, **obj_myeon, **obj_can}

# 각 카테고리의 라벨 txt 파일일의 클래스 숫자 변환
convert_label(PATH, 'dataset1_snack')
convert_label(PATH, 'dataset2_snack2')
convert_label(PATH, 'dataset3_sauce')
convert_label(PATH, 'dataset4_processed1')
convert_label(PATH, 'dataset5_processed2')
convert_label(PATH, 'dataset6_myeon')
convert_label(PATH, 'dataset7_can')


# 라벨 txt 파일을 yolov8에 맞는 형식태로 변환 후, dataset 안 train, test, val로 이미지, 텍스트 데이터 복사.
to_yolov8(PATH, 'dataset1_snack')
to_yolov8(PATH, 'dataset2_snack2')
to_yolov8(PATH, 'dataset3_sauce')
to_yolov8(PATH, 'dataset4_processed1')
to_yolov8(PATH, 'dataset5_processed2')
to_yolov8(PATH, 'dataset6_myeon')
to_yolov8(PATH, 'dataset7_can')


# obj_info로부터 각 클래스 항목들과 개수 지정 후, 모든 클래스 정보를 담고 있는 yaml 파일 생성
cls_list = list(obj_info.values()),
cls_num = len(obj_info.values())
data = {'train': 'train',
        'val': 'val',
        'test': 'test',
        'names': list(cls_list),
        'nc': cls_num}
with open(PATH + 'data.yaml', 'w') as f:
    yaml.dump(data, f)
with open(PATH + 'data.yaml', 'r') as f:
    myeon_yaml = yaml.safe_load(f)


print('끝!')
