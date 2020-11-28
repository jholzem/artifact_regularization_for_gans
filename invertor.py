from idinvert.perceptual_model import PerceptualModel
from idinvert.utils.visualizer import adjust_pixel_range
from idinvert.dnnlib import tflib
import tensorflow as tf
from idinvert.utils.visualizer import resize_image
import numpy as np

"""
Use this class to generate latent codes for stylegan given input images by applying gan inversion and optimization. The
logic is copied from gan_id_inversion.invert but generalized for any usecase where mappings from images to latent codes
are required.
Input to init:
    - E -> pretrained encoder with freezed parameters
    - G -> pretrained generator with unfreezed parameters
    - batch_size, number_of_iters defaults taken from gan_id_inversion.invert 
"""
# TODO: debug
class ImageToLatentCodeInvertor():
    def __init__(self, E, Gs, batch_size=4, number_of_iters=100):
        self.Gs = Gs
        self.E = E
        self.batch_size = batch_size
        self.number_of_iters = number_of_iters
        self.image_size = E.input_shape[2]
        self.sess = tf.get_default_session()
        self.input_shape = E.input_shape
        self.input_shape[0] = batch_size
        self.latent_shape = Gs.components.synthesis.input_shape
        self.latent_shape[0] = batch_size
        # Init members for optimization and graph building
        self.setter = None
        self.train_op = None
        self.x_rec = None
        self.wp = None
        x, x_255, x_rec_255, wp_enc = self.build_graph()
        self.configure_optimization(x, x_255, x_rec_255, wp_enc)
        # taken from gan_id_inversion.invert
        self.loss_weight_feat = 5e-5
        self.loss_weight_enc = 2.0
        self.learning_rate = 0.01

    def build_graph(self):
        x = tf.placeholder(tf.float32, shape=self.input_shape, name='real_image')
        x_255 = (tf.transpose(x, [0, 2, 3, 1]) + 1) / 2 * 255
        wp = tf.get_variable(shape=self.latent_shape, name='latent_code')
        self.x_rec = self.Gs.components.synthesis.get_output_for(wp, randomize_noise=False)
        x_rec_255 = (tf.transpose(self.x_rec, [0, 2, 3, 1]) + 1) / 2 * 255
        # Initialization of optimization as encoder output
        w_enc = self.E.get_output_for(x, is_training=False)
        wp_enc = tf.reshape(w_enc, self.latent_shape)
        self.setter = tf.assign(wp, wp_enc)
        return x, x_255, x_rec_255, wp_enc

    def configure_optimization(self, x, x_255, x_rec_255, wp_enc):
        perceptual_model = PerceptualModel([self.image_size, self.image_size], False)
        # semantic level loss / perceptual loss
        x_feat = perceptual_model(x_255)
        x_rec_feat = perceptual_model(x_rec_255)
        loss_feat = tf.reduce_mean(tf.square(x_feat - x_rec_feat), axis=[1])
        # pixel-domain loss
        loss_pix = tf.reduce_mean(tf.square(x - self.x_rec), axis=[1, 2, 3])
        # domain regularizer
        w_enc_new = self.E.get_output_for(self.x_rec, is_training=False)
        wp_enc_new = tf.reshape(w_enc_new, self.latent_shape)
        loss_enc = tf.reduce_mean(tf.square(self.wp - wp_enc_new), axis=[1, 2])
        # sum all losses
        loss = loss_pix + self.loss_weight_feat * loss_feat + self.loss_weight_enc * loss_enc
        # define objects for optimization
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(loss, var_list=[self.wp])
        tflib.init_uninitialized_vars()

    # Assumes inputs as:
    #   - N X H X W X C, where C=3 because of RGB and N number of images to be inverted.
    def image_to_latent_code(self, images):
        number_of_images = images.shape[0]
        latent_codes_enc = []
        latent_codes = []
        for img_idx in range(0, number_of_images, self.batch_size):
            batch = images[img_idx:img_idx + self.batch_size, :, :, :]
            # TODO: resize
            # Reshape it to shape of encoder input.
            batch = np.transpose(batch, [0, 3, 1, 2])
            # Make sure that images are in domain [-1, 1]
            batch = batch.astype(np.float32) / 255 * 2.0 - 1.0

            # Run encoder.
            self.sess.run([self.setter], {x: batch})
            outputs = self.sess.run([self.wp, self.x_rec])
            latent_codes_enc.append(outputs[0][0:batch.shape[0]])
            outputs[1] = adjust_pixel_range(outputs[1])

            # Optimize latent codes.
            col_idx = 3
            for step in range(1, self.num_iterations):
                self.sess.run(self.train_op, {x: batch})

            outputs = self.sess.run([self.wp, self.x_rec])
            latent_codes.append(outputs[0][0:batch.shape[0]])
            # TODO: concatenate list to tensor

            return latent_codes




