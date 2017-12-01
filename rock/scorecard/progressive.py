# -*- coding: utf-8 -*-
# __author__ = "chao.fang"
from __future__ import unicode_literals, print_function, division
import sys
import time


class Progressive(object):
    def __init__(self, total, step=1):
        self.total = total  # 总数据量
        self.percent1 = self.total / 100.0  # 每1%的数据量
        self.now = 0  # 当前进度
        self.step = step  # 显示的步长
        self.prcent_total = int(100 / self.step)

    def bar(self, num, msg=''):
        num += 1
        num = int(num / self.percent1) if num != self.total else 100
        prcent_now = int(num / self.step)
        if num == 100:
            self._print(msg, "=" * prcent_now, " " * (self.prcent_total - prcent_now), num)
            sys.stdout.write('\n')
        elif self.now < prcent_now:
            self.now = prcent_now
            self._print(msg, "=" * prcent_now, " " * (self.prcent_total - prcent_now), num)

    @staticmethod
    def _print(a, b, c, d):
        r = '\r{0}:[{1}{2}]{3}%'.format(a, b, c, d)
        sys.stdout.write(r)
        sys.stdout.flush()


if __name__ == '__main__':
    pg = Progressive(500, step=3)  # step表示每隔多少进度刷新一次
    for i in range(500):
        time.sleep(0.01)
        pg.bar(i, "测试1")
    print("测试结束")
    pg = Progressive(300, step=1)  # step表示每隔多少进度刷新一次
    for i in range(300):
        time.sleep(0.01)
        pg.bar(i, "测试2")
    print("测试结束")
