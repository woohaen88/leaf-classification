import tensorflow as tf

import os
import json
import settings

args = settings.args

conv_base = tf.keras.applications.ResNet50(weights="imagenet",
                                           include_top=False,
                                           input_shape=(150, 150, 3))
conv_base.summary()

from tensorflow.keras import layers, models
from imutils.paths import list_images

num_classes = len(os.listdir(args['train_dir']))

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(num_classes, activation="sigmoid"))
model.summary()

print("conv_base를 동결하기 전 훈련되는 가중치의 수: {}".format(len(model.trainable_weights)))

conv_base.trainable = False
print("conv_base를 동결한 후 훈련되는 가중치의 수: {}".format(len(model.trainable_weights)))

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers, losses, metrics

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    args['train_dir'],
    target_size=(150, 150),
    batch_size=args["batch_size"],
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    args['val_dir'],
    target_size=(150, 150),
    batch_size=args["batch_size"],
    class_mode='categorical')

model.compile(loss=losses.categorical_crossentropy,
              optimizer=optimizers.Adam(learning_rate=2e-5),
              metrics=[metrics.categorical_accuracy])

number_of_train_image = len(list(list_images(args["train_dir"])))
number_of_val_image = len(list(list_images(args["val_dir"])))
steps_per_epoch = number_of_train_image // args["batch_size"]
validation_steps = number_of_val_image // args["batch_size"]


def train(model, train_generator, validation_generator):
    history = model.fit(train_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=args["epochs"],
                        validation_data=validation_generator,
                        validation_steps=validation_steps)
    return history, model


# import sys
# if __name__ == "__main__":
#     device_name = tf.test.gpu_device_name()
#     print("----------------------------------------------------")
#     print("args: ", args)
#     try:
#         if device_name == '/device:GPU:0':
#             print('Found GPU at: {}'.format(device_name))

#         if sys.argv[1] == 'run':
#             history = train(model, train_generator, validation_generator)
#             print("[INFO] model save")
#             model.save("leaf_classification_pretrained_v1.h5")
#             print("Done")

#     except Exception as e:
#         if device_name != '/device:GPU:0':
#             print('Found CPU')

if __name__ == "__main__":
    device_name = tf.test.gpu_device_name()
    print('Found GPU at: {}'.format(device_name))
    history = train(model, train_generator, validation_generator)
    print("[INFO] model save")
    model.save("leaf_classification_pretrained_v1.h5")
    print("Done")
