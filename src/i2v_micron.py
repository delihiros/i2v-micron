import json
import warnings
import numpy as np
from scipy.ndimage import zoom
from skimage.transform import resize
from chainer import Variable
from chainer.functions import average_pooling_2d, sigmoid
from chainer.functions.caffe import CaffeFunction

import util

class I2V_micron():

    def __init__(self, net, tags=None, threshold=None):
        self.net = net
        if tags is not None:
            self.tags = np.array(tags)
            self.index = {t: i for i, t in enumerate(tags)}
        else:
            self.tags = None

        if threshold is not None:
            self.threshold = threshold
        else:
            self.threshold = None
        self.mean = np.array([ 164.76139251,  167.47864617,  181.13838569])

    def _convert_image(self, image):
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 2:
            ret = np.empty((arr.shape[0], arr.shape[1], 3), dtype=np.float32)
            ret[:] = arr.reshape(arr.shape[0], arr.shape[1], 1)
            return ret
        elif arr.ndim == 3:
            return arr[:,:,:3]
        else:
            raise TypeError('unsupported image specified')

    def _estimate(self, images):
        assert(self.tags is not None)
        imgs = [self._convert_image(img) for img in images]
        prob = self._extract(imgs, layername='prob')
        prob = prob.reshape(prob.shape[0], -1)
        return prob

    def estimate_specific_tags(self, images, tags):
        prob = self._estimate(images)
        return [{t: float(prob[i, self.index[t]]) for t in tags}
                for i in range(prob.shape[0])]

    def model(self):
        return self.net

    def resize_image(self, im, new_dims, interp_order = 1):
        if im.shape[-1] == 1 or im.shape[-1] == 3:
            im_min, im_max = im.min(), im.max()
            if im_max > im_min:
                im_std = (im - im_min) / (im_max - im_min)
                resized_std = resize(im_std, new_dims, order=interp_order)
                resized_im = resized_std * (im_max - im_min) + im_min
            else:
                ret = np.empty((new_dims[0], new_dims[1], im.shape[-1]), dtype=np.float32)
                ret.fill(im_min)
                return ret
        else:
            scale = tuple(np.array(new_dims) / np.array(im.shape[:2]))
            resized_im = zoom(im, scale + (1,), order=interp_order)
        return resized_im.astype(np.float32)

    def _forward(self, inputs, layername):
        shape = [len(inputs), 224, 224, 3]
        input_ = np.zeros(shape, dtype=np.float32)
        for ix, in_ in enumerate(inputs):
            input_[ix] = self.resize_image(in_, shape[1:])
        input_ = input_[:, :, :, ::-1]
        input_ -= self.mean
        input_ = input_.transpose((0, 3, 1, 2))
        x = Variable(input_)
        y, = self.net(inputs={'data': x}, outputs=[layername], train=False)
        return y

    def _extract(self, inputs, layername):
        if layername == 'prob':
            h = self._forward(inputs, layername='conv6_4')
            h = average_pooling_2d(h, ksize=7)
            y = sigmoid(h)
            return y.data
        elif layername == 'encode1neuron':
            h = self._forward(inputs, layername='encode1')
            y = sigmoid(h)
            return y.data
        else:
            y = self._forward(inputs, layername)
            return y.data

def make_i2v_micron(param_path, tag_path=None, threshold_path=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        net = CaffeFunction(param_path)

        kwargs = {}
        if tag_path is not None:
            tags = json.loads(open(tag_path, 'r').read())
            kwargs['tags'] = tags

        if threshold_path is not None:
            fscore_threshold = np.load(threshold_path)['threshold']
            kwargs['threshold'] = fscore_threshold

    return I2V_micron(net, **kwargs)
