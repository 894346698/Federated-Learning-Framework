# -*- coding: UTF-8 -*-
'''
超大整数超大次幂然后对超大的整数取模
(base ^ exponent) mod n
'''
import time

def exp_mode(base, exponent, n):
    bin_array = bin(exponent)[2:][::-1]#bin() 返回一个整数 int 或者长整数 long int 的二进制表示。
    r = len(bin_array)
    base_array = []

    pre_base = base
    base_array.append(pre_base)

    for _ in range(r - 1):
        next_base = (pre_base * pre_base) % n #次幂变连乘
        base_array.append(next_base)
        pre_base = next_base

    a_w_b = __multi(base_array, bin_array, n)
    return a_w_b % n


def __multi(array, bin_array, n):
    result = 1
    for index in range(len(array)):
        a = array[index]
        if not int(bin_array[index]):
            continue
        result *= a
        result = result % n  # 每一步求模，减少数据长度，加快连乘的速度
    return result


