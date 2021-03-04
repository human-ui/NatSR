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

    def __call__(self, inp, output_path=None):
        if isinstance(inp, (str, Path)):
            im = imageio.imread(inp)
            ims = im[None,:,:,:]
        else:
            ims = np.array(inp)

        ims = ims.astype(np.float32)
        if np.max(ims) > 1:
            ims /= 255

        outims = self.sess.run(self.output, feed_dict={self.input: ims})
        outims = np.uint8(np.round(np.clip(outims * 255, 0, 255)))

        if output_path is not None and len(outims) == 1:
            imageio.imsave(output_path, outims[0])
        return outims


if __name__ == '__main__':
    Upsampler()('imageio:astronaut.png', 'astronaut4x.png')
