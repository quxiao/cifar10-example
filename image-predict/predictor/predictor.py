import os
import mxnet as mx
import numpy as np
from collections import namedtuple
from bottle import route, run
import urllib
from urlparse import urlparse

Batch = namedtuple('Batch', ['data'])

custom_model_dir="/workspace/model"
custom_label_file="/workspace/model/label.txt"


class ImagePredictor (object):
    def __init__(self):
        self.Batch = namedtuple('Batch', ['data'])
        self.model_dir = "model"
        self.data_dir = "data"
        self.mod = None
        self.labels = []

    def load_model(self):
        print "init model"
        path='http://data.mxnet.io/models/imagenet/'
        [mx.test_utils.download(path+'resnet/18-layers/resnet-18-0000.params', fname='weight.params', dirname=self.model_dir),
        mx.test_utils.download(path+'resnet/18-layers/resnet-18-symbol.json', fname='deploy.symbol.json', dirname=self.model_dir),
        mx.test_utils.download(path+'synset.txt', fname='labels.csv', dirname=self.model_dir)]

        print "set cpu/gpu model"
        # set the context on CPU, switch to GPU if there is one available
        ctx = mx.cpu()

        print "loading model"
        sym, arg_params, aux_params = self._load_model()
        self.mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        self.mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], 
                label_shapes=self.mod._label_shapes)
        self.mod.set_params(arg_params, aux_params, allow_missing=True)
        print "loading labels"
        with open(os.path.join(self.model_dir, 'labels.csv'), 'r') as f:
            self.labels = [l.rstrip() for l in f]

    def _load_model(self):
        sym = mx.symbol.load(os.path.join(self.model_dir, 'deploy.symbol.json'))
        save_dict = mx.ndarray.load(os.path.join(self.model_dir, 'weight.params'))
        arg_params = {}
        aux_params = {}
        for k, v in save_dict.items():
            tp, name = k.split(':', 1)
            if tp == 'arg':
                arg_params[name] = v
            if tp == 'aux':
                aux_params[name] = v
        return sym, arg_params, aux_params

    def get_image(self, uri):
        # download and show the image
        # fname = mx.test_utils.download(url, dirname=self.data_dir)
        # img = mx.image.imread(fname)
        if urlparse(uri) == None:
            # TODO, read base64
            return None
        img = mx.image.imdecode(urllib.urlopen(uri).read())
        if img is None:
            return None
        # convert into format (batch, RGB, width, height)
        img = mx.image.imresize(img, 224, 224) # resize
        img = img.transpose((2, 0, 1)) # Channel first
        img = img.expand_dims(axis=0) # batchify
        return img

    def predict(self, uri, topN=5):
        img = self.get_image(uri)
        # compute the predict probabilities
        self.mod.forward(Batch([img]))
        prob = self.mod.get_outputs()[0].asnumpy()
        # print the top-5
        prob = np.squeeze(prob)
        a = np.argsort(prob)[::-1]
        ret = []
        for i in a[0:topN]:
            print('probability=%f, class=%s' %(prob[i], self.labels[i]))
            ret.append({'prob': float(prob[i]), 'label': self.labels[i]})
        return ret

