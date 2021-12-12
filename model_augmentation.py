import os
from tensorflow.keras import layers
from tensorflow.keras import models
from imutils.paths import list_images
from pathlib import Path

import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

args = {
    'batch_size' : 256
}
num_classes = len(os.listdir('./datasets/train'))


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(num_classes, activation='sigmoid'))

print('-------------- Model Summary --------------')
print(model.summary())

from tensorflow.keras import optimizers, losses, metrics

model.compile(loss=losses.categorical_crossentropy,
              optimizer=optimizers.Adam(learning_rate=1e-4),
              metrics=[metrics.categorical_accuracy])

# ImageDataGenerator를 사용하 디렉터리에서 이미지 읽기
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
    rotation_range=20, # 랜덤하게 회전시킬 각도
    width_shift_range=0.1, # 수평으로 평행이동
    height_shift_range=0.1, # 수직으로 평행이동
    shear_range=0.1, # 전단변환
    zoom_range=0.1, # 사진확대
    horizontal_flip=True, # 이미지를 수평으로 뒤집음
    fill_mode='nearest' # 새롭게 생성해야할 픽셀
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_dir = './datasets/train'
validation_dir = './datasets/val'
test_dir = './datasets/test'

train_generator = train_datagen.flow_from_directory(
    train_dir, # 타깃 디렉터리
    target_size=(150, 150), # 모든 이미지를 150x150 크기로 바꿈
    batch_size=args['batch_size'],
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=args['batch_size'],
    class_mode='categorical'
)

train_image_num = len(list(list_images(train_dir)))
test_image_num = len(list(list_images(test_dir)))

steps_per_epoch = train_image_num // args['batch_size']
validation_steps = test_image_num // args['batch_size']

def baseline_model(model):
    history = model.fit(train_generator,
                        steps_per_epoch=steps_per_epoch, 
                        epochs=30,
                        validation_data=validation_generator,    validation_steps=validation_steps)
    return history


import sys
if __name__ == '__main__':
    if sys.argv[1] == 'run':
        print(device_lib.list_local_devices())
        history = baseline_model(model)
        print('----------------------- [INFO] MODEL SAVE --------------------')    
        current_path = Path(__file__).resolve().parent
        print('current Path: current_path')
        model.save('leaf_classification_argumentation_v2.h5')        