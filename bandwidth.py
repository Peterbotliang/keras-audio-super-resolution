#!/usr/bin/env python
# coding: utf-8

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
print(tf.__version__)
print(tf.keras.__version__)
print(tf.__path__)
import numpy as np

from tqdm import tqdm, tqdm_notebook
from utils import emphasis
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Sequence
import librosa
import librosa.display

print(tf.test.is_gpu_available())


# ## SRCNN
class SubPixel1D(tf.keras.layers.Layer):
    def __init__(self, r=2):
        super(SubPixel1D, self).__init__()
        self.r = r
    def call(self, I):
        """One-dimensional subpixel upsampling layer
        Calls a tensorflow function that directly implements this functionality.
         We assume input has dim (batch, width, r)
        """

        X = tf.transpose(I, [2,1,0]) # (r, w, b)
        X = tf.batch_to_space_nd(X, [self.r], [[0,0]]) # (1, r*w, b)
        X = tf.transpose(X, [2,1,0])
        return X

noisy = tf.keras.layers.Input(shape=(None, 1))
x_input = noisy
x = x_input

# B = 8
# n_filters = [128, 256, 512, 512, 512, 512, 512, 512]
# kernel_sizes = [65, 33, 17, 9, 9, 9, 9, 9]

B = 4
n_filters = [128, 256, 512, 512]
kernel_sizes = [65, 33, 17, 9]

# B = 3
# n_filters = [128, 256, 512]
# kernel_sizes = [65, 33, 17]

# B = 3
# n_filters = [64, 128, 256]
# kernel_sizes = [65, 33, 17]


# Downsampling Layers
encoder_features = []
for k, n_filter, kernel_size in zip(range(B), n_filters, kernel_sizes):
    x = tf.keras.layers.Conv1D(filters = n_filter,
                               kernel_size = kernel_size,
                               strides = 2,
                               padding = 'same',
                               kernel_initializer = 'Orthogonal')(x)
    # x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    encoder_features.append(x)
    
# Bottleneck Layer
x = tf.keras.layers.Conv1D(filters = 512,
                           kernel_size = 9,
                           strides = 2,
                           padding = 'same',
                           kernel_initializer = 'Orthogonal')(x)
x = tf.keras.layers.Dropout(rate=0.5)(x)
# x = tf.keras.layers.PReLU()(x)
x = tf.keras.layers.LeakyReLU(0.2)(x)

# Upsampling Layer
for k, n_filter, kernel_size, enc in reversed(list(zip(range(B), 
                                                  n_filters, 
                                                  kernel_sizes, 
                                                  encoder_features))):
    x = tf.keras.layers.Conv1D(filters = 2 * n_filter,
                               kernel_size = kernel_size,
                               strides = 1,
                               padding = 'same',
                               kernel_initializer = 'Orthogonal')(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    # x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.ReLU()(x)
    x = SubPixel1D()(x)
    x = tf.keras.layers.Concatenate(axis=2)([x, enc])

# Final Conv Layer
x = tf.keras.layers.Conv1D(filters = 2,
                           kernel_size = 9,
                           strides = 1,
                           padding = 'same')(x)
x = SubPixel1D()(x)
x_final = tf.keras.layers.Add()([x, x_input])    
G = tf.keras.models.Model(inputs = [noisy], outputs = [x_final])    


# Train Model
# Initialize Model

optim = tf.keras.optimizers.Adam(lr=1e-4)
def G_loss(true, fake):
    return K.mean(K.sqrt(K.mean((fake - true) ** 2 + 1e-6, axis=[1, 2])), axis=0)

def G_LSD_loss(y_clean, y_noisy):
    y_clean = tf.squeeze(y_clean)
    y_noisy = tf.squeeze(y_noisy)
    
    D_clean = tf.signal.stft(signals = y_clean,
                             frame_length = 2048,
                             frame_step = 1024)
    D_noisy = tf.signal.stft(signals = y_noisy,
                             frame_length = 2048,
                             frame_step = 1024)
    
    D_clean_log = K.log(K.abs(D_clean) ** 2 + 1e-6)
    D_noisy_log = K.log(K.abs(D_noisy) ** 2 + 1e-6)

	return K.mean(K.sqrt(K.mean((D_clean_log - D_noisy_log) ** 2, axis = [2])), axis = [0, 1])

G.compile(loss = G_LSD_loss,
          optimizer = optim)
G.summary()
# tf.keras.utils.plot_model(G, to_file='./generator.png', show_shapes=True)


# Training

class data_sequence(Sequence):
    def __init__(self, data_path, batch_size = 64):
        self.filenames = [os.path.join(data_path, filename) for filename in os.listdir(data_path)]
        self.batch_size = batch_size
            
    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))
    
    def on_epoch_end(self):
        np.random.shuffle(self.filenames)

    def __getitem__(self, idx):
        noisy_batch = []
        clean_batch = []
        
        for i in range(idx * self.batch_size, min(len(self.filenames), (idx + 1) * self.batch_size)):
            pair = np.load(self.filenames[i])
            # pair = emphasis(pair[np.newaxis, :, :], emph_coeff=0.95).reshape(2, -1)
            clean = pair[0].reshape(-1, 1).astype('float32')
            noisy = pair[1].reshape(-1, 1).astype('float32')
    
            noisy_batch.append(noisy)
            clean_batch.append(clean)

        return np.array(noisy_batch), np.array(clean_batch)
        
                
train_data_path = '../dataset/serialized_train_data'
val_data_path = '../dataset/serialized_val_data'    
    
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='./model/weights_LSD.hdf5', 
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True),
    tf.keras.callbacks.TensorBoard(log_dir='./logs/LSD', update_freq='batch'),
    # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-8),
             ]
    
G.fit_generator(generator = data_sequence(train_data_path, 64),
                validation_data = data_sequence(val_data_path, 2),
                steps_per_epoch = 3325 // 64, 
                verbose = 1,
                epochs = 400,
                callbacks = callbacks,
                max_queue_size = 10,
                use_multiprocessing = True,
                workers = 6,
                initial_epoch = 0)




