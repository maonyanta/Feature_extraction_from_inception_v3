import tensorflow as tf
import numpy as np

#IMG_PATH = 'path/to/input/image.jpg'
#MODEL_PATH = 'path/to/classify_image_graph_def.pb'
IMG_PATH = './panda.jpg'
MODEL_PATH = '../classify_image_graph_def.pb'


# Load pre-trained model
inception_v3 = tf.gfile.FastGFile(MODEL_PATH, 'rb')
graph_def =tf.GraphDef()
graph_def.ParseFromString(inception_v3.read())
tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Specify layer
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')

    # Reading input image
    image_data = tf.gfile.FastGFile(IMG_PATH, 'rb').read()

    # Extracting features
    features = sess.run(pool3, {'DecodeJpeg/contents:0': image_data})
    # For png image, 'DecodeJpeg:0' should be set.
    # features = sess.run(pool3, {'DecodeJpeg:0': image_data})
    print(np.squeeze(features))
