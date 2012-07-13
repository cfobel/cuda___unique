from __future__ import division
import copy
import math
import random

from nose.tools import eq_, ok_, nottest
import numpy as np

from ..unique.cuda import unique_inplace


@nottest
def test_unique_cuda_single(gold_data, data, thread_count):
    cuda_data = unique_inplace(data, thread_count=thread_count)
    cuda_data.sort()
    #if not (cuda_data == gold_data).all():
    print 'cuda_data:', cuda_data
    print 'gold_data:', gold_data
    if len(cuda_data) == len(gold_data):
        ok_((cuda_data == gold_data).all())
    else:
        eq_(cuda_data, gold_data)


@nottest
def test_unique_cuda(data_size, thread_count=None):
    print '[test_unique_cuda] %d' % data_size

    np.random.seed(0)
    data = np.arange(data_size, dtype=np.int32)
    data = np.resize(data, 2 * data.size)
    print data
    np.random.shuffle(data)
    gold_data = np.array(sorted(list(set(data))))

    if thread_count:
        thread_counts = [thread_count]
    else:
        thread_counts = [1, 3, 7, 32, 33, 46, 129, 138]

    for thread_count in thread_counts:
        yield test_unique_cuda_single, gold_data, data, thread_count


def test_unique_cuda_all():
    random.seed(0)

    for n in [13, 78, 123, 128, 256, 1024, 2048]:
        for t in test_unique_cuda(n):
            yield t
