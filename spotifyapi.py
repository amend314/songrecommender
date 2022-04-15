import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import credentials
import pandas as pd

scope = "user-top-read"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=credentials.client_id,
                                               client_secret=credentials.client_secret,
                                               scope=scope, redirect_uri='http://localhost:8888/callback'))

playlists = sp.current_user_playlists()
playlist_id = (playlists['items'][0]['id'])


def getTrackId():

    song_id = []

    p_songs = sp.playlist(playlist_id, additional_types=('track',))
    p_songs = p_songs['tracks']['items']

    for x in p_songs:
        song_id.append(x['track']['id'])

    song_id = ['spotify:track:' + s for s in song_id]

    return song_id


def getTrackName():

    tracks = []
    songs = sp.playlist(playlist_id, additional_types=('track', ))['tracks']['items']

    for x in songs:
        tracks.append(x['track']['name'])

    return tracks


def getArtistName():

    artist = []
    songs = sp.playlist(playlist_id, additional_types=('track', ))['tracks']['items']

    for x in songs:
        i = (x['track']['artists'])
        for y in i:
            artist.append(y['name'])

    return artist


def buildDf():
    data = {'trackid': getTrackId(), 'artist_name': getArtistName(), 'track_name': getTrackName(), 'pid': 99999}
    df2 = pd.DataFrame(data=data)
    df2['trackindex'] = df2['trackid'].astype('category').cat.codes

    df2['num_holdouts'] = pd.NaT
    df2['trackindex'] = df2['trackid'].astype('category').cat.codes
    df2.drop_duplicates()

    return df2
