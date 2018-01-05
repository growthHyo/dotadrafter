import sqlite3
import numpy as np
from model import get_model, save_model
import time

MAX_HEROES = 200
BATCH_SIZE = 6400

conn = sqlite3.connect('data/pub_matches.db')
c = conn.cursor()

model = get_model()

def match_to_matrix(match):
    matrix = np.zeros(shape=(MAX_HEROES * 2), dtype=np.int8)
    for r_hero in match[0].split(','):
        matrix[int(r_hero)] = 1
    for d_hero in match[1].split(','):
        matrix[int(d_hero) + MAX_HEROES] = 1
    return matrix

def epoch():
    t1 = time.time()
    iteration = 0
    c.execute('select count(*) from match')
    iterations = round(c.fetchall()[0][0] / BATCH_SIZE)
    metrics = np.zeros(len(model.metrics_names), dtype=np.float)
    c.execute('select radiant_heroes, dire_heroes, radiant_win from match order by random()')
    while True:
        data = c.fetchmany(BATCH_SIZE)
        if (len(data) == 0):
            print(metrics/iteration)
            break
        batch_x = np.zeros(shape=(len(data), MAX_HEROES * 2))
        batch_y = np.zeros(shape=(len(data), 2))
        for i, m in enumerate(data):
            batch_x[i] = match_to_matrix(m)
            batch_y[i] = [1,0] if m[2] else [0,1]
        
        metrics += model.train_on_batch(batch_x, batch_y)
        iteration += 1
        if iteration % 50 == 0:
            print(str(round(iteration / iterations * 100, 2)) + '%', metrics / iteration)
            t2 = time.time()
            t1 = t2

def test():
    iteration = 0
    metrics = np.zeros(len(model.metrics_names), dtype=np.float)
    while True:
        c.execute('select radiant_heroes, dire_heroes, radiant_win from val_match limit ? offset ?', (BATCH_SIZE, BATCH_SIZE * iteration))
        data = c.fetchall()
        if (len(data) == 0):
            print(metrics/iteration)
            break
        batch_x = np.zeros(shape=(len(data), MAX_HEROES * 2))
        batch_y = np.zeros(shape=(len(data), 2))
        for i, m in enumerate(data):
            batch_x[i] = match_to_matrix(m)
            batch_y[i] = [1,0] if m[2] else [0,1]
        
        metrics += model.test_on_batch(batch_x, batch_y)
        iteration += 1

for i in range(20):
    print('EPOCH', i)
    epoch()
    print('TEST', i)
    test()
    save_model(model)
