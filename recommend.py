from keras.models import load_model
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import os
import spotifyapi

csvList = []

for files in os.listdir('data'):
    df = pd.read_csv('data/' + files, sep=',', header=0)
    csvList.append(df)

df = pd.concat(csvList, axis=0, ignore_index=True)
df = pd.concat([df, spotifyapi.buildDf()], axis=0, ignore_index=True)
df['num_holdouts'] = pd.NaT
df['trackindex'] = df['trackid'].astype('category').cat.codes
df.drop_duplicates()

desired_user_id = 99999
model_path = 'spotify_NCF_8_[64, 32, 16, 8].h5'
print('using model: %s' % model_path)
model = load_model(model_path)
print('Loaded model!')

mlp_user_embedding_weights = (next(iter(filter(lambda x: x.name == 'mlp_user_embedding', model.layers))).get_weights())

# get the latent embedding for your desired user
user_latent_matrix = mlp_user_embedding_weights[0]
one_user_vector = user_latent_matrix[desired_user_id,:]
one_user_vector = np.reshape(one_user_vector, (1,32))

print('\nPerforming kmeans to find the nearest users/playlists...')
# get 100 similar users
kmeans = KMeans(n_clusters=100, random_state=0, verbose=0).fit(user_latent_matrix)
desired_user_label = kmeans.predict(one_user_vector)
user_label = kmeans.labels_
neighbors = []
for user_id, user_label in enumerate(user_label):
    if user_label == desired_user_label:
        neighbors.append(user_id)
print('Found {0} neighbor users/playlists.'.format(len(neighbors)))

# get the tracks in similar users' playlists
tracks = []
for user_id in neighbors:
    tracks += list(df[df['pid'] == int(user_id)]['trackindex'])
print('Found {0} neighbor tracks from these users.'.format(len(tracks)))

users = np.full(len(tracks), desired_user_id, dtype='int32')
items = np.array(tracks, dtype='int32')

print('\nRanking most likely tracks using the NeuMF model...')
# and predict tracks for my user
results = model.predict([users,items],batch_size=100, verbose=0)
results = results.tolist()
print('Ranked the tracks!')


results_df = pd.DataFrame(np.nan, index=range(len(results)), columns=['probability','track_name', 'track artist'])
print(results_df.shape)

# loop through and get the probability (of being in the playlist according to my model), the track, and the track's artist
for i, prob in enumerate(results):
    results_df.loc[i] = [prob[0], df[df['trackindex'] == i].iloc[0]['track_name'], df[df['trackindex'] == i].iloc[0]['artist_name']]
results_df = results_df.sort_values(by=['probability'], ascending=False)

print(results_df.head(20))
