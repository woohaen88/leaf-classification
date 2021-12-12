# 최초 한번 실행
import os
import shutil

original_dataset_dir = './datasets/Plant_leave_diseases_dataset_without_augmentation' # 원본 데이터셋이 위치한 경로
classes_list = os.listdir(original_dataset_dir) # 해당 경로 하위에 있는 모든 폴더의 목록을 가져옴
base_dir = './datasets'

classes_list.remove('.DS_Store')

def makedirs(path):
    if not os.path.isdir(path):
        print('create Directory : {}'.format(path))
        os.makedirs(path)

    else:
        print('Directory exist')

# 나눈 데이터를 저장할 폴더를 생성
makedirs(base_dir)

# 분리 후에 각 데이터를 저장할 하위 폴더 train, val, test를 생성
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

dirs = [train_dir, validation_dir, test_dir]
list(map(makedirs, dirs))

# train, validation, test 폴더 하위에 각각 클래스 목록 폴더를 생성
for clss in classes_list:
    makedirs(os.path.join(train_dir, clss))
    makedirs(os.path.join(validation_dir, clss))
    makedirs(os.path.join(test_dir, clss))

import math
for clss in classes_list:
    # path 위치에 존재하는 모든 이미지 파일의 목록을 fnames에 저장
    path = os.path.join(original_dataset_dir, clss)
    fnames = os.listdir(path)

    # Train, Validation, Test 데이터의 비율을 지정(6:2:2)
    train_size = math.floor(len(fnames) * 0.6)
    validation_size = math.floor(len(fnames) * 0.2)
    test_size = math.floor(len(fnames) * 0.2)

    # Train 데이터에 해당하는 파일의 이름을 train_fnames에 저장
    train_fnames = fnames[:train_size]
    print('Train Size({}): {}'.format(clss, len(train_fnames)))
    # 모든 Train 데이터에 대해 for 문의 내용을 반복
    for fname in train_fnames:
        src = os.path.join(path, fname)
        dst = os.path.join(train_dir, clss, fname)
        shutil.copyfile(src, dst)

    # Validation 데이터에 해당하는 파일의 이름을 validation_fnames에 저장
    validation_fnames = fnames[train_size:(validation_size+train_size)]
    print('Validaation Size({}): {}'.format(clss, len(validation_fnames)))
    # 모든 Validation 데이터에 대해 for 문의 내용을 반복
    for fname in validation_fnames:
        src = os.path.join(path, fname)
        dst = os.path.join(validation_dir, clss, fname)
        shutil.copyfile(src, dst)

    # Train 데이터에 해당하는 파일의 이름을 train_fnames에 저장
    test_fnames = fnames[(train_size+validation_size):(validation_size+train_size+test_size)]
    print('Test Size({}): {}'.format(clss, len(test_fnames)))
    # 모든 Train 데이터에 대해 for 문의 내용을 반복
    for fname in test_fnames:
        src = os.path.join(path, fname)
        dst = os.path.join(test_dir, clss, fname)
        shutil.copyfile(src, dst)