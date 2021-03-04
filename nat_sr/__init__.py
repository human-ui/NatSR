import os
from pathlib import Path

import numpy as np
import imageio
import tensorflow.compat.v1 as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.disable_eager_execution()


class Upsampler:

    def __init__(self):
        saver = tf.train.import_meta_graph('resources/NatSR.meta')
        self.input, self.output = tf.get_collection('InNOut')
        self.sess = tf.Session()
        saver.restore(self.sess, 'resources/NatSR')

    def __call__(self, ims):
        ims = np.array(ims).astype(np.float32)
        if ims.ndim == 3:
            ims = ims[None,:,:,:]
        if np.max(ims) > 1:
            ims /= 255

        outims = self.sess.run(self.output, feed_dict={self.input: ims})
        outims = np.uint8(np.round(np.clip(outims * 255, 0, 255)))

        return outims


if __name__ == '__main__':
    ims = imageio.imread('imageio:astronaut.png')
    out = Upsampler()(ims)
    imageio.imwrite('astronaut4x.png', out[0])
