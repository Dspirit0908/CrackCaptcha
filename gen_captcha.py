# coding: utf-8

import random
import numpy as np
from setting import *
from PIL import Image
import matplotlib.pyplot as plt
from captcha.image import ImageCaptcha  # pip install captcha


def random_captcha_text(captcha_size, char_set=CHAR_SET_FOR_GEN_CAPTCHA):
    """
    :param char_set: 验证码字符来源
    :param captcha_size: 验证码长度
    :return: 随机的验证码文本 list
    """
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


def gen_captcha_text_and_image(captcha_size):
    """
    :return: 生成随机验证码文本及其对应的图像
    """
    captcha_list = random_captcha_text(captcha_size)
    captcha_text = ''.join(captcha_list)

    captcha = ImageCaptcha().generate(captcha_text)
    # ImageCaptcha().write(captcha_text, captcha_text + '.jpg')  # 写到文件

    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image


if __name__ == '__main__':
    text, image = gen_captcha_text_and_image(4)

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    plt.imshow(image)
    plt.show()
