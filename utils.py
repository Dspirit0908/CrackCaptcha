# coding: utf-8

import os
import cv2
import random
import shutil
import numpy as np
from setting import *
from PIL import Image
from pylab import array
from gen_captcha import gen_captcha_text_and_image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


def binarization(image_gray):
    """
    采用cv2的自适应二值化算法（领域内均值）后，进行0/1转换
    :param image_gray: 灰度图
    :return: 0/1二值化后的图像
    """
    image_binary = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 10)

    for i in range(image_binary.shape[0]):
        for j in range(image_binary.shape[1]):
            if image_binary[i][j] == 255:
                image_binary[i][j] = 0
            elif image_binary[i][j] == 0:
                image_binary[i][j] = 1

    return image_binary


def convert2gray(image):
    if len(image.shape) == 3:
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    elif len(image.shape) == 1:
        return image
    else:
        print('Warning: The number of channels is %d, can not be converted to grayscale.' % image.shape)
        return image


def data_augmentation(pics_dir, max_captcha, image_data_generator=None):
    if isinstance(image_data_generator, ImageDataGenerator):
        data_generator = image_data_generator
    else:
        data_generator = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,
            fill_mode='nearest',
        )

    images_list = os.listdir(pics_dir)
    for image_file_name in images_list:
        image_file = os.path.join(pics_dir, image_file_name)
        image = load_img(image_file)
        x = img_to_array(image)  # this is a Numpy array with shape (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)

        i = 0
        for batch in data_generator.flow(x, batch_size=1, save_to_dir=pics_dir,
                                         save_prefix=image_file_name[0:max_captcha], save_format=image_file_name[-3:]):
            i += 1
            if i >= 9:
                break  # 生成9张新的图片


def get_next_batch(max_captcha, image_height, image_width, batch_size=64):
    """
    自动生成验证码
    :param batch_size: 每batch的图片数量
    :return: 一个完整的batch，包括batch_x和batch_y
    """
    batch_x = np.zeros([batch_size, image_height * image_width])
    batch_y = np.zeros([batch_size, max_captcha * CHAR_SET_LEN])

    # 有时生成图像大小不是(60, 160, 3)
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image(max_captcha)
            if image.shape == (60, 160, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image_gray = convert2gray(image)

        batch_x[i, :] = image_gray.flatten() / 255
        batch_y[i, :] = text2vec(text, max_captcha)

    return batch_x, batch_y


def get_next_image(pics_dir, max_captcha, image_height, image_width, batch_size=64):
    image_list = os.listdir(pics_dir)
    random.shuffle(image_list)

    batch_x = np.zeros([batch_size, image_height * image_width])
    batch_y = np.zeros([batch_size, max_captcha * CHAR_SET_LEN])

    for i in range(batch_size):
        image_file_name = image_list[i]
        image_file = os.path.join(pics_dir, image_file_name)
        image = array(Image.open(image_file).resize((image_width, image_height)))
        image_gray = convert2gray(image)

        batch_x[i, :] = image_gray.flatten() / 255
        batch_y[i, :] = text2vec(image_file_name[0:max_captcha], max_captcha)

    return batch_x, batch_y


def split(images_dir, train_dir, test_dir, train_num=None, ratio=0.9):
    images_list = os.listdir(images_dir)
    random.shuffle(images_list)

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    if train_num is None:
        train_num = int(len(images_list) * ratio)

    for image_file in images_list[0:train_num]:
        shutil.copy(os.path.join(images_dir, image_file), os.path.join(train_dir, image_file))

    for image_file in images_list[train_num:]:
        shutil.copy(os.path.join(images_dir, image_file), os.path.join(test_dir, image_file))


def text2vec(text, max_captcha):
    """
    文本转向量
    """
    if len(text) > max_captcha:
        raise ValueError('验证码最长%d个字符' % max_captcha)
    vector = np.zeros(max_captcha * CHAR_SET_LEN)
    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2index[c]
        vector[idx] = 1
    return vector


def vec2text(vec):
    """
    向量转回文本
    """
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_idx = c % CHAR_SET_LEN
        text.append(index2char[char_idx])
    return ''.join(text)


if __name__ == '__main__':
    data_generator = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest',
    )
    data_augmentation('./train', 4, data_generator)
    pass
