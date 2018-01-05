import numpy as np
from peewee import *
import os
from model import get_model, save_model, reload_model
import tensorflow as tf

db_path = 'data/doto.db'
weights_path = 'data/model.ckpt'

db = SqliteDatabase(db_path)

class Match(Model):
    id = IntegerField(primary_key=True)
    seq_num = IntegerField()
    radiant_heroes = CharField()
    dire_heroes = CharField()
    radiant_win = BooleanField()
    class Meta:
        database = db
        
class Hero(Model):
    id = IntegerField(primary_key=True)
    name = CharField()
    class Meta:
        database = db
        
db.connect()
        
heroes = dict()
        
for h in Hero.select():
    heroes[h.id] = h.name.replace("_", " ")
    
hero_translations = {
    'shadow fiend' : 'nevermore',
    'natures prophet' : 'furion',
    'timber saw' : 'shredder',
    'clockwerk' : 'rattletrap',
    'zeus' : 'zuus',
    'io' : 'wisp',
    'outworld devourer' : 'obsidian destroyer',
    'windranger' : 'windrunner',
    'necrophos' : 'necrolyte',
    'wraith king' : 'skeleton king'
}

max_heroes = 200
n_input = max_heroes * 2
test_batch = 1500

mmr_scale = np.load('data/mmr_scale.npy')
mmr_default = 4000

def softmax(w, scale=1):
    w = np.array(w)
    maxes = np.amax(w, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    e = np.exp(w - maxes)
    dist = e / np.sum(e, axis=1, keepdims=True) * scale
    return dist

class DotoAnn:
    def __init__(self):
        self.heroes = heroes
        self.hero_translations = hero_translations
        self.model = get_model()
        self.graph = tf.get_default_graph()
            
    def save(self):
        save_model(self.model)
        
    def reload(self):
        reload_model(self.model)
        
    def run(self, inp, mmr=mmr_default):
        mmr_offset = mmr_scale * (mmr - mmr_default)
        r_offset = inp.dot(np.vstack((mmr_offset, np.zeros((max_heroes, 1))))) + 1
        d_offset = inp.dot(np.vstack((np.zeros((max_heroes, 1)), mmr_offset))) + 1
        offset = np.hstack((r_offset, d_offset))
        out = None
        with self.graph.as_default():
            out = self.model.predict(inp)
        out = np.multiply(out, offset)
        return softmax(out*2.1)
    
    def get_hero_id(self, hero, r_heroes=[], d_heroes=[]):
        if hero and len(hero) > 0:
            for id,name in heroes.items():
                if hero in name:
                    if id not in r_heroes and id not in d_heroes:
                        return id
            for name in hero_translations:
                if hero in name:
                    found = self.get_hero_id(hero_translations[name])
                    if found:
                        return found
        return None
        
    def queryDraft(self, r_heroes_str, d_heroes_str, mmr=mmr_default):
        r_heroes = []
        d_heroes = []

        print_actual_winrate = False

        for hero in r_heroes_str:
            id = self.get_hero_id(hero, r_heroes, d_heroes)
            if id:
                r_heroes.append(id)
                    
        for hero in d_heroes_str:
            id = self.get_hero_id(hero, r_heroes, d_heroes)
            if id:
                d_heroes.append(id)
                    
        #hero_win_rates = dict()
        #for h in heroes:
        #    inp = np.zeros((1, n_input), np.int)
        #    inp[0][h] = 1
        #    hero_win_rates[h] = self.run(inp)[0][0]
        #    h += max_heroes
        #    inp = np.zeros((1, n_input), np.int)
        #    inp[0][h] = 1
        #    hero_win_rates[h] = self.run(inp)[0][1]

        inp = np.zeros((n_input+1, n_input), np.int)
        for h in r_heroes:
            inp[:, int(h)] = np.ones(n_input+1)
        for h in d_heroes:
            inp[:, int(h) + max_heroes] = np.ones(n_input+1)

        for h in heroes:
            if h not in r_heroes and h not in d_heroes:
                inp[h][h] = 1
                inp[h+max_heroes][h + max_heroes] = 1
            elif h in r_heroes:
                inp[h][h] = 0
            elif h in d_heroes:
                inp[h+max_heroes][h+max_heroes] = 0
                
        out = self.run(inp, mmr)
        current_ch = out[n_input]
        
        r_next = []
        d_next = []
        r_current = []
        d_current = []

        for h in heroes:
            if h not in r_heroes and h not in d_heroes:
                r_next.append({'hero':heroes[h], 'score':float(out[h][0])})
                d_next.append({'hero':heroes[h], 'score':float(out[h+max_heroes][1])})
                
        for h in r_heroes:
            r_current.append({'hero':heroes[h], 'score':float(out[h][0])})
        for h in d_heroes:
            d_current.append({'hero':heroes[h], 'score':float(out[h+max_heroes][1])})

        resp = dict()
        #resp = {'r_picks':'','d_picks':''}
        
        #for h in sorted(picks, key=picks.get, reverse=True)[:10]:
        #    resp['r_picks'] += h + "\t\t" + str(picks[h]) + "\n"
            
        #for h in sorted(counters, key=counters.get, reverse=True)[:10]:
        #    resp['d_picks'] += h + "\t\t" + str(counters[h]) + "\n"

        if print_actual_winrate:
            query_winrate = Match.select()
            for h in r_heroes:
                query_winrate = query_winrate.where(
                    (Match.radiant_heroes ** (str(h) + ",%")) | 
                    (Match.radiant_heroes ** ("%," + str(h) + ",%")) | 
                    (Match.radiant_heroes ** ("%," + str(h)))
                )
            for h in d_heroes:
                query_winrate = query_winrate.where(
                    (Match.dire_heroes ** (str(h) + ",%")) | 
                    (Match.dire_heroes ** ("%," + str(h) + ",%")) | 
                    (Match.dire_heroes ** ("%," + str(h)))
                )
            team1_wins = query_winrate.where(Match.radiant_win == True).count()
            team2_wins = query_winrate.where(Match.radiant_win == False).count()

            query_winrate = Match.select()
            for h in d_heroes:
                query_winrate = query_winrate.where(
                    (Match.radiant_heroes ** (str(h) + ",%")) | 
                    (Match.radiant_heroes ** ("%," + str(h) + ",%")) | 
                    (Match.radiant_heroes ** ("%," + str(h)))
                )
            for h in r_heroes:
                query_winrate = query_winrate.where(
                    (Match.dire_heroes ** (str(h) + ",%")) | 
                    (Match.dire_heroes ** ("%," + str(h) + ",%")) | 
                    (Match.dire_heroes ** ("%," + str(h)))
                )
            team1_wins += query_winrate.where(Match.radiant_win == False).count()
            team2_wins += query_winrate.where(Match.radiant_win == True).count()
            resp['actual'] = dict()
            resp['actual']['sample'] = team1_wins + team2_wins
            if team1_wins + team2_wins > 0:
                resp['actual']['winrate'] = team1_wins / (team1_wins + team2_wins)
            
        resp['prediction'] = current_ch.tolist()
        resp['heroes'] = dict()
        resp['heroes']['r_next'] = r_next
        resp['heroes']['d_next'] = d_next
        resp['heroes']['r_current'] = r_current
        resp['heroes']['d_current'] = d_current
        return resp
        
    def train(self, xs, ys):
        self.model.train_on_batch(xs, ys)

    def test_accuracy(self, xs=None, ys=None):
        if xs is None:
            xs = np.zeros((test_batch, n_input), np.int)
            ys = np.zeros((test_batch, 2), np.int)
            i=0
            #for m in Match.select().order_by(fn.Random()).limit(test_batch):
            for m in Match.select().order_by(Match.seq_num.desc()).limit(test_batch):
                for h in m.radiant_heroes.split(","):
                    xs[i][int(h)] = 1
                for h in m.dire_heroes.split(","):
                    xs[i][int(h) + max_heroes] = 1
                ys[i][0 if m.radiant_win else 1] = 1
                
                #for h in m.radiant_heroes.split(","):
                #    xs[i][int(h) + max_heroes] = 1
                #for h in m.dire_heroes.split(","):
                #    xs[i][int(h)] = 1
                #ys[i][1 if m.radiant_win else 0] = 1
                i+=1
        return self.model.test_on_batch(xs, ys)
    