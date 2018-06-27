#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import os,io
import sys
import urllib
from glob2 import glob
import csv
import json
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import image
import cPickle

from collections import OrderedDict # 有序的词典


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def save_image(data, basedir, filenames):
    for d, f in zip(data, filenames):
        img = d.reshape(3, 32, 32)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 0, 1)
        assert img.shape == (32,32,3)
        # matplotlib.image.imsave(f, img)
        image.imsave(os.path.join(basedir, f), img)

def parse_data(file):
    print 'parsing file %s' % (file)
    test = unpickle(file)
    data = test['data']
    labels = test['labels']
    batch_label = test['batch_label']
    filenames = test['filenames']

    # batches.meta
    label_names = unpickle(os.path.dirname(file) + '/batches.meta')['label_names']
    print 'label_names: %s' % label_names
    print 'save images'
    save_image(data, 'data/image', filenames)
    print 'done'


def main():
    for i in range(5):
        parse_data('data/cifar-10-batches-py/data_batch_%d' % (i+1))

if __name__ == '__main__':
    main()