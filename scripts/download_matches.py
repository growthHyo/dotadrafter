import requests
import urllib
import random
import sqlite3

conn = sqlite3.connect('../data/data.db')
c = conn.cursor()
val_split = 0.15

sql = """SELECT
matches.match_id,
matches.start_time,
matches.radiant_win,
player_matches.player_slot,
player_matches.hero_id
FROM matches
JOIN player_matches using(match_id)
WHERE TRUE
ORDER BY matches.match_id DESC
"""

api_url = "https://api.opendota.com/api/explorer?sql="
req_url = api_url + urllib.parse.quote_plus(sql)

res = requests.get(req_url)
rows = res.json()['rows']

matches = {}

for row in rows:
    if not row['match_id'] in matches:
        matches[row['match_id']] = {
            'id': row['match_id'],
            'radiant_win': row['radiant_win'],
            'radiant_heroes': [],
            'dire_heroes': []
        }
    team = 'radiant_heroes' if row['player_slot'] < 128 else 'dire_heroes'
    matches[row['match_id']][team].append(row['hero_id'])
    
print(len(matches))

c.execute('DELETE FROM match')
c.execute('DELETE FROM val_match')

for m in matches.values():
    r_heroes = ",".join(str(hero_id) for hero_id in m['radiant_heroes'])
    d_heroes = ",".join(str(hero_id) for hero_id in m['dire_heroes'])
    table = 'match'
    if (random.random() < val_split):
        table = 'val_match'
    c.execute('INSERT INTO ' + table + ' (id, radiant_heroes, dire_heroes, radiant_win) VALUES (?, ?, ?, ?)',
             (m['id'], r_heroes, d_heroes, m['radiant_win']))
    
conn.commit()
conn.close()
