import tensorflow as tf
from tensorflow.keras import layers

class Generator(tf.keras.Model):

    def __init__(self):
        super(Generator, self).__init__(name = 'Generator')

        self.enc1 = tf.keras.layers.Conv1D(filters = 16,
                                           kernel_size = 32,
                                           strides = 2,
                                           padding = 'same')
        self.enc1_act = tf.keras.layers.PReLU()
        self.enc2 = tf.keras.layers.Conv1D(32, 32, 2, 'same')
        self.enc2_act = tf.keras.layers.PReLU()
        self.enc3 = tf.keras.layers.Conv1D(32, 32, 2, 'same')
        self.enc3_act = tf.keras.layers.PReLU()
        self.enc4 = tf.keras.layers.Conv1D(64, 32, 2, 'same')
        self.enc4_act = tf.keras.layers.PReLU()
        self.enc5 = tf.keras.layers.Conv1D(64, 32, 2, 'same')
        self.enc5_act = tf.keras.layers.PReLU()
        self.enc6 = tf.keras.layers.Conv1D(128, 32, 2, 'same')
        self.enc6_act = tf.keras.layers.PReLU()
        self.enc7 = tf.keras.layers.Conv1D(128, 32, 2, 'same')
        self.enc7_act = tf.keras.layers.PReLU()
        self.enc8 = tf.keras.layers.Conv1D(256, 32, 2, 'same')
        self.enc8_act = tf.keras.layers.PReLU()
        self.enc9 = tf.keras.layers.Conv1D(256, 32, 2, 'same')
        self.enc9_act = tf.keras.layers.PReLU()
        self.enc10 = tf.keras.layers.Conv1D(512, 32, 2, 'same')
        self.enc10_act = tf.keras.layers.PReLU()
        self.enc11 = tf.keras.layers.Conv1D(1024, 32, 2, 'same')
