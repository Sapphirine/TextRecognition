import numpy as np
import scipy.io as sio
import os
from PIL import Image

image_path = "result/"


def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def crop_image(image_input):
    # crop the input image with size 16*512, turn it to 63 image with 16*16 (store as 64*256)
    image_length = image_input.shape[0]
    image_stride = int(image_length / 2)
    image_num = int(2 * image_input.shape[1] / image_length)
    # our output here is 64*256
    image_output = np.zeros((image_num, image_length * image_length), dtype='int32')

    for i in range(image_num - 1):
        array = np.reshape(image_input[:, i * image_stride:i * image_stride + image_length], newshape=(-1,))
        image_output[i, :] = array

    new_array = np.hstack((image_input[:, -image_stride:], image_input[:, :image_stride]))
    image_output[-1, :] = new_array.reshape(-1, )

    return image_output


def parse_data(training_num=500, validation_num=100, test_num=100):
    print("parsing image into mat data...")
    check_path("data/")

    set_size = [training_num, validation_num, test_num]
    set_name = ["training", "validation", "test"]

    start_index = 0

    for turn in range(3):
        num = set_size[turn]

        input_0 = np.asarray(Image.open((image_path + "line_%d.png") % start_index).convert('L'), dtype='int32')
        input = crop_image(input_0)

        output_0 = np.asarray(Image.open((image_path + "line_truth_%d.png") % start_index).convert('L'), dtype='int32')
        output = crop_image(output_0)

        for i in range(start_index + 1, start_index + num):
            train_image = np.asarray(Image.open((image_path + "line_%d.png") % i).convert('L'), dtype='int32')
            test_image = np.asarray(Image.open((image_path + "line_truth_%d.png") % i).convert('L'), dtype='int32')

            # print(train_image)
            train_crop = crop_image(train_image)
            test_crop = crop_image(test_image)

            input = np.vstack((input, train_crop))
            output = np.vstack((output, test_crop))

        start_index += num

        # store the data into mat_file
        file_name = "data/" + set_name[turn] + ".mat"
        print("... saving mat " + file_name)
        sio.savemat(file_name, {"set_input": input, "set_output": output})


def load_data():
    train_file_name = "data/training.mat"
    valid_file_name = "data/validation.mat"
    test_file_name = "data/test.mat"

    print("... load data")

    train_set = sio.loadmat(train_file_name)
    valid_set = sio.loadmat(valid_file_name)
    test_set = sio.loadmat(test_file_name)

    train_set = (train_set['set_input'], train_set['set_output'])
    valid_set = (valid_set['set_input'], valid_set['set_output'])
    test_set = (test_set['set_input'], test_set['set_output'])

    rval = [train_set, valid_set, test_set]

    return rval


def reform_image(image_matrix, image_length, name_index=0, recover_path="recover_result/"):
    # recover the image with the correspond cropping method, the image_length is the edge length of image
    # we constrain our input as 63*(16*16) -__-
    assert image_matrix.shape[0] == 64 and image_matrix.shape[1] == image_length * image_length

    image_stride = int(image_length / 2)
    recover_img = np.zeros((image_length, image_length * 32))

    for i in range(63):
        array = image_matrix[i].reshape((image_length, image_length))
        recover_img[:, i * image_stride:i * image_stride + image_length] += array

    last_row = image_matrix[-1].reshape((image_length, image_length))
    recover_img[:image_length, -image_stride:] += last_row[:, :image_stride]
    recover_img[:image_length, :image_stride] += last_row[:, -image_stride:]
    recover = np.asarray(recover_img / 2, dtype=np.uint8)

    print("#%d image of test result successfully!" % name_index)
    img = Image.fromarray(recover.astype(np.uint8))
    img.save(recover_path + 'test_result_#%d.png' % name_index)
    return recover


def batch_recover(batch_input, image_length=16, rows_in_single_image=64, recover_path="recover_result/"):
    assert batch_input.shape[0] % rows_in_single_image == 0

    check_path(recover_path)

    turns = int(batch_input.shape[0] / rows_in_single_image)

    out_image = []
    for i in range(turns):
        input_matrix = batch_input[i * rows_in_single_image:(i + 1) * rows_in_single_image, :]
        recover = reform_image(image_matrix=input_matrix,
                               image_length=image_length,
                               name_index=i,
                               recover_path=recover_path)
        if i == 0:
            out_image = recover
        else:
            out_image = np.vstack((out_image, recover))

    img = Image.fromarray(out_image.astype(np.uint8))
    img.save(recover_path + "batch_result.png")


if __name__ == '__main__':
    parse_data(200, 50, 50)
    rval = load_data()
    train_input, train_output = rval[0]
    valid_input, valid_output = rval[1]
    test_input, test_output = rval[2]

    # batch_recover(train_output, image_length=16, rows_in_single_image=64)
