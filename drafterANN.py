import numpy as np
from peewee import *
import os
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

mmr_scale = np.load('data/mmr_scale.npy')
mmr_default = 4000
    
# Parameters
learning_rate = 0.00005
epoch_size = 1500
max_heroes = 120

# Network Parameters
n_hidden_1 = max_heroes * 10
n_hidden_2 = max_heroes * 10
n_input = max_heroes * 2
n_out = 2

# Placeholders
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_out])
layer_opac = tf.placeholder("float")

# Create model
def multilayer_perceptron(_X, _weights, _biases, _layer_opac):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['s1']), _biases['b1']))
    layer_1 = tf.nn.dropout(layer_1, _layer_opac)
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['s2']), _biases['b2']))
    layer_2 = tf.nn.dropout(layer_2, _layer_opac)
    return tf.add(tf.matmul(layer_2, _weights['s3']), _biases['b3'])

# Weight & bias
weights = {
    's1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=0.06, mean=0.0), name='weights1'),
    's2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=0.06, mean=0.0), name='weights2'),
    's3': tf.Variable(tf.random_normal([n_hidden_2, n_out], stddev=0.06, mean=0.0), name='weights3')
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], stddev=0.002, mean=0.01), name='biases1'),
    'b2': tf.Variable(tf.random_normal([n_hidden_2], stddev=0.002, mean=0.01), name='biases2'),
    'b3': tf.Variable(tf.random_normal([n_out], stddev=0.002, mean=0.01), name='biases3')
}


# Define model operations
pred = multilayer_perceptron(x, weights, biases, layer_opac)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
init = tf.initialize_all_variables()
saver2 = tf.train.Saver(list(weights.values()) + list(biases.values()))
saver = tf.train.Saver(tf.all_variables())

def softmax(w, scale=1):
    w = np.array(w)
    maxes = np.amax(w, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    e = np.exp(w - maxes)
    dist = e / np.sum(e, axis=1, keepdims=True) * scale
    return dist

sess = tf.Session()

class DotoAnn:
    def __init__(self): 
        self.heroes = heroes
        self.hero_translations = hero_translations
        if os.path.isfile(weights_path):
            sess.run(init)
            saver.restore(sess, weights_path)
        else:
            print("Weights file not found")
            sess.run(init)
            
    def save(self):
        saver.save(sess, weights_path)
        
    def reload(self):
        saver.restore(sess, weights_path)
        
    def run(self, inp, mmr=mmr_default):
        mmr_offset = mmr_scale * (mmr - mmr_default)
        r_offset = inp.dot(np.vstack((mmr_offset, np.zeros((max_heroes, 1))))) + 1
        d_offset = inp.dot(np.vstack((np.zeros((max_heroes, 1)), mmr_offset))) + 1
        offset = np.hstack((r_offset, d_offset))
        out = sess.run(tf.nn.softmax(pred), feed_dict={x: inp, layer_opac: 1})
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

        for h in heroes:
            if h not in r_heroes and h not in d_heroes:
                inp[h+max_heroes][h + max_heroes] = 1
                
        out = self.run(inp, mmr)
        current_ch = out[n_input]
        
        picks = []
        counters = []

        for h in heroes:
            if h not in r_heroes and h not in d_heroes:
                picks.append({'hero':heroes[h], 'score':float(out[h][0])})
                counters.append({'hero':heroes[h], 'score':float(out[h+max_heroes][1])})

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
        resp['heroes']['r'] = [heroes[h] for h in r_heroes]
        resp['heroes']['d'] = [heroes[h] for h in d_heroes]
        resp['heroes']['r_next'] = picks
        resp['heroes']['d_next'] = counters
        return resp
        
    def train(self, xs, ys):
        sess.run(optimizer, feed_dict={x: xs, y: ys, layer_opac: 0.99})
        
    def test_accuracy(self, xs=None, ys=None):
        if xs is None:
            xs = np.zeros((epoch_size, n_input), np.int)
            ys = np.zeros((epoch_size, n_out), np.int)
            i=0
            #for m in Match.select().order_by(fn.Random()).limit(epoch_size):
            for m in Match.select().order_by(Match.seq_num.desc()).limit(epoch_size):
                for h in m.radiant_heroes.split(","):
                    xs[i][int(h)] = 1
                for h in m.dire_heroes.split(","):
                    xs[i][int(h) + max_heroes] = 1
                ys[i][not m.radiant_win] = 1
                i+=1
        return sess.run(accuracy, feed_dict={x: xs, y: ys, layer_opac: 1})
    