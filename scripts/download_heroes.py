import sqlite3
import requests

api_key = 'FE70CE9FC0D6D99279498CE852587F59'

conn = sqlite3.connect('../data/data.db')
c = conn.cursor()

c.execute('DELETE FROM hero')

r = requests.get('https://api.steampowered.com/IEconDOTA2_570/GetHeroes/V001/?key=' + api_key)

for hero in r.json()['result']['heroes']:
    c.execute('INSERT INTO hero (id, name) VALUES (?, ?)', (hero['id'], hero['name'][14:]))

for hero in c.execute('SELECT * FROM hero'):
    print(hero)

conn.commit()
conn.close()
