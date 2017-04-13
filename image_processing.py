import numpy as np
from PIL import Image, ImageEnhance
import os
import re


def increase_contrast(image_name):
    image_file = Image.open(image_name)  # open colour image
    image_file = image_file.convert('L')  # convert image to monochrome - this works
    image_file = ImageEnhance.Sharpness(image_file).enhance(3)
    image_file = ImageEnhance.Contrast(image_file).enhance(2)

    threshold = 80
    image_file = image_file.point(lambda p: p > threshold and 255)

    # degree = np.random.randint(low=-10, high=10)
    # image_file = rotate_img(image_file, degree)

    # image_file.save('result.png')
    image_file.save(image_name)


def rotate_img(image, degree):
    # the image here is an image object
    size = image.size

    new_img = Image.Image.rotate(image, degree, expand=True)
    new_img.resize(size)
    return new_img


if __name__ == '__main__':
    in_path = "result/"

    dirs = os.listdir(in_path)
    regex = r"[t]"
    for img in dirs:
        if not re.search(regex, img):
            print(img)
            increase_contrast(in_path+img)

