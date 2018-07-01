import os
import base64
import mxnet as mx
import numpy as np
from collections import namedtuple
from bottle import route, run
import urllib
from urlparse import urlparse

Batch = namedtuple('Batch', ['data'])
base64_data_prefix = 'data:application/octet-stream;base64,'


class ImagePredictor (object):
    def __init__(self):
        self.Batch = namedtuple('Batch', ['data'])
        self.model_dir = "model"
        self.data_dir = "data"
        self.mod = None
        self.labels = []
        self.symbol_filename = 'deploy.symbol.json'
        self.weight_filename = 'weight.params'
        self.label_filename = 'labels.csv'

    def load_model(self, custom_model_dir=None, symbol_fn=None, weight_fn=None, label_fn=None):
        if custom_model_dir != None:
            self.model_dir = custom_model_dir
        if symbol_fn != None:
            self.symbol_filename = symbol_fn
        if weight_fn != None:
            self.weight_filename = weight_fn
        if label_fn != None:
            self.label_filename = label_fn

        print "set cpu/gpu model"
        # set the context on CPU, switch to GPU if there is one available
        ctx = mx.cpu()

        print "loading model"
        sym, arg_params, aux_params, labels = self._load_model()
        self.mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        self.mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], 
                label_shapes=self.mod._label_shapes)
        self.mod.set_params(arg_params, aux_params, allow_missing=True)
        self.labels = labels

    def _load_model(self):
        sym = mx.symbol.load(os.path.join(self.model_dir, self.symbol_filename))
        save_dict = mx.ndarray.load(os.path.join(self.model_dir, self.weight_filename))
        arg_params = {}
        aux_params = {}
        for k, v in save_dict.items():
            tp, name = k.split(':', 1)
            if tp == 'arg':
                arg_params[name] = v
            if tp == 'aux':
                aux_params[name] = v

        labels = []
        with open(os.path.join(self.model_dir, self.label_filename), 'r') as f:
            labels = [l.rstrip() for l in f]

        return sym, arg_params, aux_params, labels

    def get_image(self, uri):
        # download and show the image
        # fname = mx.test_utils.download(url, dirname=self.data_dir)
        # img = mx.image.imread(fname)
        if urlparse(uri) == None and uri.startswith(base64_data_prefix):
            content = base64.decodestring(uri[len(base64_data_prefix):])
        elif urlparse(uri) != None:
            content = urllib.urlopen(uri).read()
        else:
            return None
        img = mx.image.imdecode(content)
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
        print 'uri: %s' % (uri)
        for i in a[0:topN]:
            print('probability=%f, class=%s' %(prob[i], self.labels[i]))
            ret.append({'prob': float(prob[i]), 'label': self.labels[i]})
        print ''
        return ret

