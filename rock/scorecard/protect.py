# -*- coding: utf-8 -*-
# __author__ = "chao.fang"
from __future__ import unicode_literals, print_function, division
import getpass
import socket

__all__ = ['check_apply']


def check_apply():
    users = ['chao.fang', 'xiaotong.jiang', 'dong.liu', 'wenchao.yin', 'wei.ye',
             'shuo.gan', 'huarui.wang', 'hongwei.ren', 'jianwei.zhang',
             'jiaqi.wang', 'lang.hu', 'lei.shi', 'lingyi.dai', 'yongqiang.wu']
    # hosts = ['spark-p-039045.hz.td', 'spark-p-039026.hz.td',
    #          'ml-gpu-p-053133.hz.td', 'ml-gpu-p-053135.hz.td']
    user = getpass.getuser()
    host = socket.gethostname()
    if user not in users:
        raise SystemError("system error!")
    if 'hz.td' not in host:
        raise SystemError("system error!")


if __name__ == '__main__':
    check_apply()
