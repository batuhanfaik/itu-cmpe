import os
import re
import shutil

import cv2
import numpy as np
from skimage import exposure


def _is_image(file):
    file = file.lower()
    if file.endswith((".jpeg", ".png", ".jpg")):
        return True
    else:
        return False


class Preprocessor:
    def __init__(self, dataset_path, crx_params, mode="c", new_dataset_name="preprocessed_dataset"):
        assert type(dataset_path) == str
        assert type(crx_params) == dict
        assert type(mode) == str and len(mode) == 1, \
            "Mode must be a char. 'c': copy, 'o': overwrite"
        self.dataset_path = dataset_path
        self.crx_params = crx_params
        self.mode = mode
        self.new_dataset_name = new_dataset_name

        self.means = []
        self.stds = []

    def _image_process(self, file, root):
        full_path = os.path.join(root, file)
        # Read grayscale
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        # Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=self.crx_params["clip_limit"],
                                tileGridSize=self.crx_params["tile_grid_size"])
        img = clahe.apply(img)
        # Median Filtering
        img = cv2.medianBlur(img, self.crx_params["median_filter_size"])
        # Contrast Stretching
        lower_percentile = np.percentile(img, self.crx_params["percentiles"][0])
        upper_percentile = np.percentile(img, self.crx_params["percentiles"][1])
        img = exposure.rescale_intensity(img, in_range=(lower_percentile, upper_percentile))
        return img

    def preprocess_dataset(self):
        dataset_root = re.search(r".*\/", self.dataset_path)[0]
        new_dataset_path = self.dataset_path
        if self.mode == "c":
            new_dataset_path = os.path.join(dataset_root, self.new_dataset_name)

        # Get all files and preprocess
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                set_folder = root.split("/")[-1]
                if _is_image(file):
                    img = self._image_process(file, root)
                    # Do not peek into the test folder
                    if set_folder != "test":
                        mean, std = cv2.meanStdDev(img)
                        self.means.append(mean[0][0])
                        self.stds.append(std[0][0])

                    if self.mode == "c":
                        if not os.path.exists(os.path.join(dataset_root, self.new_dataset_name, set_folder)):
                            os.makedirs(os.path.join(dataset_root, self.new_dataset_name, set_folder))
                        save_dir = os.path.join(dataset_root, self.new_dataset_name, set_folder, file)
                    elif self.mode == "o":
                        save_dir = os.path.join(self.dataset_path, set_folder, file)
                    else:
                        save_dir = ""
                        assert RuntimeError, "Preprocessing mode is not valid."

                    cv2.imwrite(save_dir, img)
                elif self.mode == "c":     # Copy files beside images
                    if not os.path.exists(os.path.join(dataset_root, self.new_dataset_name, set_folder)):
                        os.makedirs(os.path.join(dataset_root, self.new_dataset_name, set_folder))
                    save_dir = os.path.join(dataset_root, self.new_dataset_name, set_folder, file)
                    shutil.copy(os.path.join(root, file), save_dir)

        dataset_mean = np.around(np.mean(self.means), decimals=0).astype(np.uint8)
        dataset_std = np.around(np.mean(self.stds), decimals=0).astype(np.uint8)
        print("Preprocessing completed successfully\nNew Mean: {}\nNew STD: {}".format(dataset_mean, dataset_std))
        return new_dataset_path, dataset_mean, dataset_std
