import os
import shutil
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataStructure:

    def __init__(self, working_dir, data_vals: tuple, main_dir="/"):
        os.chdir(working_dir)
        self.main_dir = main_dir
        self.train_dir = self.main_dir + "train/"
        self.validation_dir = self.main_dir + "validation/"
        self.raw_jpg_dir = self.main_dir + "root_jpg/"
        self.data_values = data_vals

    @staticmethod
    def _create_dir(dir_to_create):
        if not os.path.exists(dir_to_create):
            os.mkdir(dir_to_create)

    @staticmethod
    def _crop_image(image_path):
        image_obj = Image.open(image_path)

        # width, height = image_obj.size

        # coord_width = [0, width]
        #
        # coord_height_diff = (height - width) / 2
        # coord_height = [coord_height_diff, height - coord_height_diff]

        # coords = (coord_width[0], coord_height[0], coord_width[1], coord_height[1])
        # image_obj = image_obj.crop(coords)
        image_obj = image_obj.rotate(270)
        image_obj.save(image_path)

    def create_file_structure(self):
        self._create_dir(self.main_dir)
        self._create_dir(self.train_dir)
        self._create_dir(self.validation_dir)

        for i in range(len(self.data_values)):
            check_path = self.data_values[i] + "/"

            self._create_dir(self.train_dir + check_path)
            self._create_dir(self.validation_dir + check_path)

    def add_img_to_structure(self, data_set_dir):

        data_set_dir = self.main_dir + data_set_dir

        try:
            data_set_list = next(os.walk(data_set_dir))[1]

            for img_case in data_set_list:
                dir_list_raw = os.listdir(data_set_dir + img_case)
                dir_list = []

                for img in dir_list_raw:
                    if img.endswith(".jpg"):
                        dir_list.append(img)

                for i in range(len(dir_list)):
                    if i < 90:
                        dir_dst = self.train_dir
                    else:
                        dir_dst = self.validation_dir

                    single_img = dir_list[i]
                    src = self.raw_jpg_dir + img_case + "/" + single_img
                    dst = dir_dst + "{}/{}.{}.jpg".format(img_case, img_case, i)
                    shutil.copy(src, dst)

                    # self._crop_image(dst)
        except StopIteration:
            print("stop iteration")

    def tensor_pre_process(self, img_height, img_width, batch_size):
        img_height = img_height
        img_width = img_width
        train_image_generator = ImageDataGenerator(rescale=1./255)
        validation_image_generator = ImageDataGenerator(rescale=1./255)

        train_data_gen = train_image_generator.flow_from_directory(
            batch_size=batch_size,
            directory=self.train_dir,
            shuffle=True,
            target_size=(img_height, img_width),
            class_mode='binary'
        )

        val_data_gen = validation_image_generator.flow_from_directory(
            batch_size=batch_size,
            directory=self.validation_dir,
            shuffle=True,
            target_size=(img_height, img_width),
            class_mode='binary'
        )

        return train_data_gen, val_data_gen

# data_set = DataStructure(
#     working_dir="/volumes/external_1/",
#     data_vals=("zero", "one", "two", "three", "four"),
#     main_dir="./numbers_jpg/"
# )
#
# data_set.create_file_structure()
#
# data_set.add_img_to_structure("root_jpg/")






