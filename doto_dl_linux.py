import datetime as dt
import time
from drafterANN import *
from dota_api import Dota_API
import sys

sys.setrecursionlimit(2000)

drafter = DotoAnn()
api = Dota_API()

next_id = '2238010584'
latest_start_time = None

for m in Match.select().order_by(Match.seq_num.desc()).limit(1):
    next_id = m.seq_num + 1

i = 0
matches_parsed = 0

req = api.matches_get(n_id=next_id)

matches_bulk = list()
batch_size = 64

try:
    while True:
        i += 1
        if i%100==1:
            print('Errors: ', api.errors)
            print('Matches: ', matches_parsed)
            api.errors = 0
            matches_parsed = 0
            print(time.strftime("%H:%M:%S") + ": " + str(latest_start_time))
            
            inp = np.zeros((1, n_input), np.int)
            inp[0][2] = 1
            print("Axe: " + str(drafter.run(inp)))
            
            inp = np.zeros((1, n_input), np.int)
            inp[0][2 + max_heroes] = 1
            print("vs Axe: " + str(drafter.run(inp)))
            
            print("")
            drafter.save()
            
        if i%500==1:
            print("Accuracy:", drafter.test_accuracy())
    
        matches = api.matches_result(req)
        
        next_id = matches[-1]['match_seq_num'] + 1
        if latest_start_time and dt.datetime.now() - latest_start_time > dt.timedelta(hours=3):
             next_id += 5000
        req = api.matches_get(n_id=next_id)
        
        reqs_dict = dict()
        matches_dict = dict()

        for match in matches:
            if match['human_players'] < 10:
                #print('skip: <10 players')
                continue
            if not (match['lobby_type'] == 7 or match['lobby_type'] == 2 or match['lobby_type'] == 5):
                #print('skip: noob lobby type')
                continue
            for player in match['players']:
                if player['leaver_status'] == 1:
                    #print('skip: abandon')
                    continue
            m_id = match['match_id']
            reqs_dict[m_id] = api.matches_get(6, m_id)
            matches_dict[m_id] = match
        for m_id, req2 in reqs_dict.items():
            skill = api.matches_result(req2)
            if (skill == 'Very High'):   
                #add match to list
                match = matches_dict[m_id]
                m = dict(id=match['match_id'], seq_num=match['match_seq_num'], radiant_win=match['radiant_win'])
                r_heroes = []
                d_heroes = []
                for player in match['players']:
                    if player['player_slot'] < 100:
                        r_heroes.append(str(player['hero_id']))
                    else:
                        d_heroes.append(str(player['hero_id']))
                m['radiant_heroes'] = ",".join(r_heroes)
                m['dire_heroes'] = ",".join(d_heroes)
                matches_bulk.append(m)
            
                #train on batch
                if len(matches_bulk) >= batch_size:
                    #save matches to db
                    with db.atomic():
                        Match.insert_many(matches_bulk).execute()

                    #train the network
                    batch_xs = np.zeros((len(matches_bulk), n_input), np.int)
                    batch_ys = np.zeros((len(matches_bulk), n_out), np.int)
                    i2=0
                    for m in matches_bulk:
                        for h in m['radiant_heroes'].split(","):
                            batch_xs[i2][int(h)] = 1
                        for h in m['dire_heroes'].split(","):
                            batch_xs[i2][int(h) + max_heroes] = 1
                        batch_ys[i2][0 if m['radiant_win'] else 1] = 1
                        i2+=1
                    drafter.train(batch_xs, batch_ys)
                    matches_bulk = list()

                    start_time = dt.datetime.fromtimestamp(match['start_time'])
                    if not latest_start_time or start_time > latest_start_time:
                        latest_start_time = start_time
                    matches_parsed += batch_size
        drafter.save()

except BaseException as e:
    raise e
finally:
    drafter.save()
    db.close()
    sess.close()
    print("Errors: ", api.errors)
    print("Matches: ", matches_parsed)
    print("Saved")
