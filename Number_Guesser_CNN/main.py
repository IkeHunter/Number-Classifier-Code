import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from pre_process import DataStructure
import os

"""Import Data Set"""

BATCH_SIZE = 20
EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 40

data_set = DataStructure(
    working_dir="/volumes/external_1/",
    data_vals=("zero", "one", "two", "three", "four"),
    main_dir="./numbers_jpg/"
)

data_set.create_file_structure()

data_set.add_img_to_structure("root_jpg/")

train_data, val_data = data_set.tensor_pre_process(img_height=IMG_HEIGHT, img_width=IMG_WIDTH, batch_size=BATCH_SIZE)


# sample_training_images, _ = next(train_data)
#
# def plotImages(images_arr):
#     fig, axes = plt.subplots(1, 5, figsize=(30,40))
#     axes = axes.flatten()
#     for img, ax in zip( images_arr, axes):
#         ax.imshow(img)
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()
#
# plotImages(sample_training_images[:5])


"""CNN Model"""
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(5, activation='softmax'))


"""Compile CNN"""
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

"""Run CNN"""
history = model.fit(train_data, epochs=EPOCHS, validation_data=val_data)

"""Plotting Accuracy"""
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')


"""Saving CNN"""
os.chdir("/Users/isaachunter/desktop/number_guesser_cnn")
os.getcwd()

save_path = "./dnn/image_classifier_model"
file_path = "{}/saved_model.pb".format(save_path)

model.save(save_path)

converter = tf.lite.TFLiteConverter.from_saved_model(save_path)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)


"""Specify model's input/output"""
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

# Input shape/type
print(interpreter.get_input_details()[0]['shape'])
print(interpreter.get_input_details()[0]['dtype'])

# Output shape/type
print(interpreter.get_output_details()[0]['shape'])
print(interpreter.get_output_details()[0]['dtype'])

