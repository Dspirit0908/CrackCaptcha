# coding: utf-8

import os
import tensorflow as tf

# 训练集位置
train_dir = r'D:\Captcha2\captcha_train'

# 测试集位置
test_dir = r'D:\Captcha2\test_images'

# tensorboard logdir 位置
logdir = './logs'

# base model 位置
base_model = tf.train.get_checkpoint_state(os.getcwd()).model_checkpoint_path

# 最大字符长度
MAX_CAPTCHA = 5

# 图片高度
IMAGE_HEIGHT = 60

# 图片宽度
IMAGE_WIDTH = 160

# 字符集
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']
CHAR_SET = number + alphabet + ALPHABET + ['_']  # 如果验证码长度小于MAX_CAPTCHA, '_'用来补齐
CHAR_SET_LEN = len(CHAR_SET)

CHAR_SET_FOR_BATCH = number + alphabet + ALPHABET  # 生成验证码时不考虑空缺，所以不需要'_'

# 字符和CHAR_SET列表下标互相转换
char2index = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'a': 10, 'b': 11, 'c': 12,
              'd': 13, 'e': 14, 'f': 15, 'g': 16, 'h': 17, 'i': 18, 'j': 19, 'k': 20, 'l': 21, 'm': 22, 'n': 23,
              'o': 24, 'p': 25, 'q': 26, 'r': 27, 's': 28, 't': 29, 'u': 30, 'v': 31, 'w': 32, 'x': 33, 'y': 34,
              'z': 35, 'A': 36, 'B': 37, 'C': 38, 'D': 39, 'E': 40, 'F': 41, 'G': 42, 'H': 43, 'I': 44, 'J': 45,
              'K': 46, 'L': 47, 'M': 48, 'N': 49, 'O': 50, 'P': 51, 'Q': 52, 'R': 53, 'S': 54, 'T': 55, 'U': 56,
              'V': 57, 'W': 58, 'X': 59, 'Y': 60, 'Z': 61, '_': 62}
index2char = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'a', 11: 'b', 12: 'c',
              13: 'd', 14: 'e', 15: 'f', 16: 'g', 17: 'h', 18: 'i', 19: 'j', 20: 'k', 21: 'l', 22: 'm', 23: 'n',
              24: 'o', 25: 'p', 26: 'q', 27: 'r', 28: 's', 29: 't', 30: 'u', 31: 'v', 32: 'w', 33: 'x', 34: 'y',
              35: 'z', 36: 'A', 37: 'B', 38: 'C', 39: 'D', 40: 'E', 41: 'F', 42: 'G', 43: 'H', 44: 'I', 45: 'J',
              46: 'K', 47: 'L', 48: 'M', 49: 'N', 50: 'O', 51: 'P', 52: 'Q', 53: 'R', 54: 'S', 55: 'T', 56: 'U',
              57: 'V', 58: 'W', 59: 'X', 60: 'Y', 61: 'Z', 62: '_'}

if __name__ == '__main__':
    pass
