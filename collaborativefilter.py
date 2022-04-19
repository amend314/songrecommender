import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import Sequential, Model, load_model, save_model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, Multiply, Reshape, Flatten, Concatenate, merge
from keras.regularizers import l2
import scipy.sparse as sp
import os
import matplotlib.pyplot as plt


csvList = []

for files in os.listdir('data'):
    df = pd.read_csv('data/' + files, sep=',', header=0)
    csvList.append(df)

df = pd.concat(csvList, axis=0, ignore_index=True)
df['num_holdouts'] = pd.NaT
df['trackindex'] = df['trackid'].astype('category').cat.codes
df.drop_duplicates()

train = df[pd.isnull(df['num_holdouts'])]

mat = sp.dok_matrix((train.shape[0], len(df['trackindex'].unique())), dtype=np.float32)
for pid, trackindex in zip(train['pid'], train['trackindex']):
    mat[pid, trackindex] = 1.0

#sp.save_npz('spotify_train_matrix.npz', mat.tocoo())


def get_model(num_users, num_items, latent_dim=8, dense_layers=[64, 32, 16, 8],
              reg_layers=[0, 0, 0, 0], reg_mf=0):
    # input layer
    input_user = Input(shape=(1,), dtype='int32', name='user_input')
    input_item = Input(shape=(1,), dtype='int32', name='item_input')

    mf_user_embedding = Embedding(input_dim=num_users, output_dim=latent_dim,
                                  name='mf_user_embedding',
                                  embeddings_initializer='RandomNormal',
                                  embeddings_regularizer=l2(reg_mf), input_length=1)
    mf_item_embedding = Embedding(input_dim=num_items, output_dim=latent_dim,
                                  name='mf_item_embedding',
                                  embeddings_initializer='RandomNormal',
                                  embeddings_regularizer=l2(reg_mf), input_length=1)
    mlp_user_embedding = Embedding(input_dim=num_users, output_dim=int(dense_layers[0] / 2),
                                   name='mlp_user_embedding',
                                   embeddings_initializer='RandomNormal',
                                   embeddings_regularizer=l2(reg_layers[0]),
                                   input_length=1)
    mlp_item_embedding = Embedding(input_dim=num_items, output_dim=int(dense_layers[0] / 2),
                                   name='mlp_item_embedding',
                                   embeddings_initializer='RandomNormal',
                                   embeddings_regularizer=l2(reg_layers[0]),
                                   input_length=1)

    mf_user_latent = Flatten()(mf_user_embedding(input_user))
    mf_item_latent = Flatten()(mf_item_embedding(input_item))
    mf_cat_latent = Multiply()([mf_user_latent, mf_item_latent])

    mlp_user_latent = Flatten()(mlp_user_embedding(input_user))
    mlp_item_latent = Flatten()(mlp_item_embedding(input_item))
    mlp_cat_latent = Concatenate()([mlp_user_latent, mlp_item_latent])

    mlp_vector = mlp_cat_latent

    for i in range(1, len(dense_layers)):
        layer = Dense(dense_layers[i],
                      activity_regularizer=l2(reg_layers[i]),
                      activation='relu',
                      name='layer%d' % i)
        mlp_vector = layer(mlp_vector)

    predict_layer = Concatenate()([mf_cat_latent, mlp_vector])
    result = Dense(1, activation='sigmoid',
                   kernel_initializer='lecun_uniform', name='result')

    model = Model(inputs=[input_user, input_item], outputs=result(predict_layer))

    return model

def get_train_samples(train_mat, num_negatives):
    user_input, item_input, labels = [], [], []
    num_user, num_item = train_mat.shape
    for(u, i) in train_mat.keys():
        user_input.append(u)
        item_input.append(i)
        labels.append(1)

        for t in range(num_negatives):
            j = np.random.randint(num_item)
            while(u, j) in train_mat.keys():
                j = np.random.randint(num_item)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels


loaded = True
verbose = 1
epochs = 10
batch_size = 256
latent_dim = 8
dense_layers = [64, 32, 16, 8]
reg_layers = [0, 0, 0, 0]
reg_mf = 0
num_negatives = 4
learning_rate = 1e-3
learner = 'adam'
dataset = 'spotify'

if loaded:
    train_mat = mat
else:
    train_mat = sp.load_npz('spotify_train_matrix.npz')

num_users, num_items = train_mat.shape

model = get_model(num_users, num_items, latent_dim, dense_layers, reg_layers, reg_mf)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy',
              metrics=['accuracy'])
print(model.summary())


user_input, item_input, labels = get_train_samples(train_mat, num_negatives)

hist = model.fit([np.array(user_input), np.array(item_input)], np.array(labels),
                 batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=True)

model_file = '%s_NCF_%d_%s.h5' % (dataset, latent_dim, str(dense_layers))
model.save(model_file, overwrite=True)

acc = hist.history['accuracy']
loss = hist.history['loss']
ep = range(1, epochs)
plt.plot(np.arange(len(acc)), acc, 'r', label='Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

loss = hist.history['loss']
ep = range(1, epochs)
plt.plot(np.arange(len(loss)), loss, 'b', label='Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
