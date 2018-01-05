import requests
import urllib
import random
import sqlite3

source_ds = sqlite3.connect('../data/doto.db')
dest_ds = sqlite3.connect('../data/pub_matches.db')
source_c = source_ds.cursor()
dest_c = dest_ds.cursor()

val_split = 0.1

dest_c.execute('DELETE FROM match')
dest_c.execute('DELETE FROM val_match')

for match in source_c.execute('select id, radiant_heroes, dire_heroes, radiant_win from match'):
    table = 'match'
    if (random.random() < val_split):
        table = 'val_match'
    dest_c.execute('INSERT INTO ' + table + ' (id, radiant_heroes, dire_heroes, radiant_win) VALUES (?, ?, ?, ?)', match)
    
dest_ds.commit()
dest_ds.close()
source_ds.close()
