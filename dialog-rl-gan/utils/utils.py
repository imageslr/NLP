# -*- coding: UTF-8 -*-

import numpy as np

def just_message(message):
    """
    将被输出的message填充为"======= Message ======"的格式
    :return: string
    """
    return (' ' + message + ' ').center(150, '=')

def softmax(x):
    prob = np.exp(x) / np.sum(np.exp(x), axis=0)
    return prob