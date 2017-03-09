#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import codecs
import os

font_type = "/Users/Fuwy/Documents/Columbia/coursework/EECS_6895_ABD/Arial.ttf"
dataset_path = "/Users/Fuwy/Documents/Columbia/coursework/EECS_6895_ABD/Datasets/E-book/"
result_path = "result/"


def read_character(file):
    try:
        string = file.read(1)
    except UnicodeDecodeError:
        string = '?'
    return string


def read_file(file, length):
    """

    :param file: the input file stream
    :param length: the length of words
    :return: the string we get from the file
    """

    result_string = ''
    str = read_character(file)
    # remove the blanks and the turning for the first time reading
    while str != '' and (str == '\n' or str == ' '):
        str = read_character(file)
    result_string += str

    # read the words
    for i in range(length):
        str = read_character(file)
        if str == '' or str == '\n':
            return result_string
        result_string += str

    # finish reading the last word
    while str != '' and str != '\n' and str != ' ':
        str = read_character(file)
        result_string += str

    return result_string


def generate_batch_image(file,
                         store_path,
                         number,
                         size=(512, 16),
                         font_type=font_type,
                         font_size=12,
                         length=80,
                         mask_portion=3,
                         mask_from_top=True
                         ):
    # print information
    print("Generation image from file:", file.name)
    print("Picture result is stored in:", store_path)
    print("Generation picture number:", number)
    print("Size of Image:", size)
    if mask_from_top:
        print("With mask from top for: 1/", mask_portion)
    else:
        print("With mask on the bottom for: 1/", mask_portion)

    if not os.path.exists(store_path):
        os.makedirs(store_path)

    # store the data info
    info_file = open(store_path + "#info.txt", "w")
    info_file.write("size:" + str(size) + "\nnumber:" + str(number))
    info_file.close()

    for i in range(number):
        # generate a single image file and the result
        image, image_true, string = generate_single_img(
            file=file,
            size=size,
            font_type=font_type,
            font_size=font_size,
            length=length,
            mask_portion=mask_portion,
            mask_from_top=mask_from_top
        )

        if string == '':
            break

        # save the image
        if i % 100 == 0:
            print("index:%d" % i)

        image.save((store_path + "line_%d.png") % i, "PNG")
        image_true.save((store_path + "line_truth_%d.png") % i, "PNG")

        # save the ground truth
        txt_file = open(store_path + "ground_truth_%d.txt" % i, "w")
        txt_file.write(string)
        txt_file.close()


def generate_single_img(file,
                        size=(800, 20),
                        font_type=font_type,
                        font_size=14,
                        length=100,
                        mask_portion=3,
                        mask_from_top=True
                        ):
    # read the string that needs to be print
    strs = read_file(file=file, length=length)
    font = ImageFont.truetype(font_type, font_size)
    font_width, font_height = font.getsize(strs)
    # size = (font_width, font_height)

    # construct drawing
    img_true = Image.new(mode="1", size=size, color=255)
    draw_true = ImageDraw.Draw(img_true)

    bg_color = random.randrange(127) + 128
    img = Image.new(mode="L", size=size, color=bg_color)
    draw = ImageDraw.Draw(img)

    # draw text on the canvas
    draw.text(xy=(0, 0), text=strs, font=font)
    draw_true.text(xy=(0, 0), text=strs, font=font)

    # draw mask for the text
    if mask_portion <= 0:
        mask_portion = font_height

    if mask_from_top:
        draw.rectangle((0, 0, font_width, font_height / mask_portion), fill=bg_color)
    else:
        draw.rectangle((0, font_height * (1 - 1 / mask_portion), font_width, font_height), fill=bg_color)

    return img, img_true, strs


if __name__ == '__main__':
    # generate_img()
    with open(dataset_path + "Complete+Prose+Works+of+Walt+Whitman.txt", "r") as file:
        # with codecs.open(dataset_path + "test.txt", "r", encoding="utf-8") as file:
        generate_batch_image(file, result_path, number=1000)

    file.close()
