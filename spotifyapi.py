import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import credentials

scope = "user-top-read"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=credentials.client_id,
                                               client_secret=credentials.client_secret,
                                               scope=scope, redirect_uri='http://localhost:8888/callback'))

results = sp.current_user_top_tracks()

for x in results['items']:
    print(x['name'])
