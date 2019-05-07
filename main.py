import tensorflow as tf
import numpy as np
import os
from utils import emphasis
import tensorflow.keras.backend as K

def _load_numpy(filename, emph_coff=0.95):
    pair = np.load(filename.decode())
    pair = emphasis(pair[np.newaxis, :, :], emph_coeff=0.95).reshape(2, -1)
    clean = pair[0].reshape(-1, 1).astype('float32')
    noisy = pair[1].reshape(-1, 1).astype('float32')
    z = np.random.randn(8, 1024).astype('float32')
    return clean, noisy

data_path = '/home/peterliang/Software/Research/SE_proj/SEGAN_pytorch/data/serialized_train_data'
filenames = [os.path.join(data_path, filename) for filename in os.listdir(data_path)]
dataset = tf.data.Dataset.from_tensor_slices(filenames)
dataset = dataset.map(lambda filename : tf.py_func(_load_numpy,
                                              [filename, 0.95],
                                              [tf.float32, tf.float32]))

dataset = dataset.batch(32)
dataset = dataset.repeat()
# iterator = dataset.make_one_shot_iterator()
# el = iterator.get_next()
# with tf.Session() as sess:
#     print(type(sess.run(el)), len(sess.run(el)))
#     print(sess.run(el)[0].shape)
#     print(sess.run(el)[1].shape)
#     # print(sess.run(el)[2].shape)
#     print(type(sess.run(el)[0]))
#     print(type(sess.run(el)[1]))
#     # print(type(sess.run(el)[2]))

# Generator

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same', activation=None):
    x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=1))(input_tensor)
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(1, strides), padding=padding, activation=activation)(x)
    x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(x, axis=1))(x)
    return x

noisy = tf.keras.layers.Input(shape=(16384, 1))
clean = tf.keras.layers.Input(shape=(16384, 1))
z = tf.keras.layers.Input(shape=(8, 1024))
x_enc1 = tf.keras.layers.Conv1D(filters = 16,
                                kernel_size = 32,
                                strides = 2,
                                padding = 'same')(noisy)
x_enc1 = tf.keras.layers.PReLU()(x_enc1)
x_enc2 = tf.keras.layers.Conv1D(32, 32, 2, 'same')(x_enc1)
x_enc2 = tf.keras.layers.PReLU()(x_enc2)
x_enc3 = tf.keras.layers.Conv1D(32, 32, 2, 'same')(x_enc2)
x_enc3 = tf.keras.layers.PReLU()(x_enc3)
x_enc4 = tf.keras.layers.Conv1D(64, 32, 2, 'same')(x_enc3)
x_enc4 = tf.keras.layers.PReLU()(x_enc4)
x_enc5 = tf.keras.layers.Conv1D(64, 32, 2, 'same')(x_enc4)
x_enc5 = tf.keras.layers.PReLU()(x_enc5)
x_enc6 = tf.keras.layers.Conv1D(128, 32, 2, 'same')(x_enc5)
x_enc6 = tf.keras.layers.PReLU()(x_enc6)
x_enc7 = tf.keras.layers.Conv1D(128, 32, 2, 'same')(x_enc6)
x_enc7 = tf.keras.layers.PReLU()(x_enc7)
x_enc8 = tf.keras.layers.Conv1D(256, 32, 2, 'same')(x_enc7)
x_enc8 = tf.keras.layers.PReLU()(x_enc8)
x_enc9 = tf.keras.layers.Conv1D(256, 32, 2, 'same')(x_enc8)
x_enc9 = tf.keras.layers.PReLU()(x_enc9)
x_enc10 = tf.keras.layers.Conv1D(512, 32, 2, 'same')(x_enc9)
x_enc10 = tf.keras.layers.PReLU()(x_enc10)
x_enc11 = tf.keras.layers.Conv1D(1024, 32, 2, 'same')(x_enc10)
c = tf.keras.layers.PReLU()(x_enc11)

# encoded = tf.keras.layers.concatenate([c, z], axis = 0)
encoded = c

x_dec10 = Conv1DTranspose(encoded,
                          filters = 512,
                          kernel_size = 32,
                          strides = 2,
                          padding = 'same')
x_dec10 = tf.keras.layers.PReLU()(x_dec10)
x_dec10_c = tf.keras.layers.concatenate([x_dec10, x_enc10], axis = 0)
x_dec9 = Conv1DTranspose(x_dec10_c, 256, 32, 2, 'same')
x_dec9 = tf.keras.layers.PReLU()(x_dec9)
x_dec9_c = tf.keras.layers.concatenate([x_dec9, x_enc9], axis = 0)
x_dec8 = Conv1DTranspose(x_dec9_c, 256, 32, 2, 'same')
x_dec8 = tf.keras.layers.PReLU()(x_dec8)
x_dec8_c = tf.keras.layers.concatenate([x_dec8, x_enc8], axis = 0)
x_dec7 = Conv1DTranspose(x_dec8_c, 128, 32, 2, 'same')
x_dec7 = tf.keras.layers.PReLU()(x_dec7)
x_dec7_c = tf.keras.layers.concatenate([x_dec7, x_enc7], axis = 0)
x_dec6 = Conv1DTranspose(x_dec7_c, 128, 32, 2, 'same')
x_dec6 = tf.keras.layers.PReLU()(x_dec6)
x_dec6_c = tf.keras.layers.concatenate([x_dec6, x_enc6], axis = 0)
x_dec5 = Conv1DTranspose(x_dec6_c, 64, 32, 2, 'same')
x_dec5 = tf.keras.layers.PReLU()(x_dec5)
x_dec5_c = tf.keras.layers.concatenate([x_dec5, x_enc5], axis = 0)
x_dec4 = Conv1DTranspose(x_dec5_c, 64, 32, 2, 'same')
x_dec4 = tf.keras.layers.PReLU()(x_dec4)
x_dec4_c = tf.keras.layers.concatenate([x_dec4, x_enc4], axis = 0)
x_dec3 = Conv1DTranspose(x_dec4_c, 32, 32, 2, 'same')
x_dec3 = tf.keras.layers.PReLU()(x_dec3)
x_dec3_c = tf.keras.layers.concatenate([x_dec3, x_enc3], axis = 0)
x_dec2 = Conv1DTranspose(x_dec3_c, 32, 32, 2, 'same')
x_dec2 = tf.keras.layers.PReLU()(x_dec2)
x_dec2_c = tf.keras.layers.concatenate([x_dec2, x_enc2], axis = 0)
x_dec1 = Conv1DTranspose(x_dec2_c, 16, 32, 2, 'same')
x_dec1 = tf.keras.layers.PReLU()(x_dec1)
x_dec1_c = tf.keras.layers.concatenate([x_dec1, x_enc1], axis = 0)
x_final = Conv1DTranspose(x_dec1_c, 1, 32, 2, 'same', activation='tanh')

G = tf.keras.Model(inputs = [clean, noisy], outputs = x_final)

optim = tf.keras.optimizers.Adam(lr=1e-4)
def G_loss(true, fake):
    def lossfun(y_true, y_pred):
        return 1 * K.mean(K.abs(fake - true))
    return lossfun

G.compile(loss = G_loss(clean, x_final),
          optimizer = optim)
G.summary()
tf.keras.utils.plot_model(G, to_file='./generator.png', show_shapes=True)

G.fit(dataset,
      steps_per_epoch=1)
