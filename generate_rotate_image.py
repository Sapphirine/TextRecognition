from PIL import Image
import numpy as np
import os


def generate_random_rotate(in_path="result/", out_path="rotate_image/"):
    if not os.path.exists(in_path):
        return

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    dirs = os.listdir(in_path)
    for img in dirs:
        if img.startswith("line_truth"):
            image = Image.open(in_path + img)

            degree = np.random.randint(low=-10, high=10)
            image_new = image.rotate(degree, expand=1)

            image_new.save(out_path + "rotate" + img)


if __name__ == '__main__':
    generate_random_rotate()