import sqlite3

conn = sqlite3.connect('../data/pub_matches.db')
c = conn.cursor()

# c.execute('''CREATE TABLE hero (id text, name text, PRIMARY KEY (id))''')
c.execute('''CREATE TABLE match (id text, radiant_heroes text, dire_heroes text, radiant_win bool, PRIMARY KEY (id))''')
c.execute('''CREATE TABLE val_match (id text, radiant_heroes text, dire_heroes text, radiant_win bool, PRIMARY KEY (id))''')

conn.commit()
conn.close()
