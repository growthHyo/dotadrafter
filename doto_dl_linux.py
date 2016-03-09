import requests
from requests_futures.sessions import FuturesSession
import datetime as dt
import time
import sys
from drafterANN import *
import socket

api_keys = ['FE70CE9FC0D6D99279498CE852587F59','2FEC67172AAC0C393EC209A225A7E51E']
api_key_num = 1
api_key = api_keys[api_key_num]
sleep_time = 7000

drafter = DotoAnn()

next_id = '1808518935'
latest_start_time = None

for m in Match.select().order_by(Match.seq_num.desc()).limit(1):
    next_id = m.seq_num + 1

i = 0
errors = 0
session = FuturesSession()

def next_matches():
    url = 'https://api.steampowered.com/IDOTA2Match_570/GetMatchHistoryBySequenceNum/V001/?key=' + api_key + '&start_at_match_seq_num=' + str(next_id) + '&min_players=10'
    return session.get(url, timeout=4)

req = next_matches()

while True:
    i += 1
    try:
        if i%1000==1:
            print('Errors: ', errors)
            errors = 0
            print(time.strftime("%H:%M:%S") + ": " + str(latest_start_time))
            inp = np.zeros((1, n_input), np.int)
            inp[0][7] = 1
            inp[0][61 + max_heroes] = 1
            print("ES vs Brood: " + str(drafter.run(inp)))

            inp = np.zeros((1, n_input), np.int)
            inp[0][57] = 1
            print("Omni: " + str(drafter.run(inp)))
            print("")

            drafter.save()
            
        if i%3000==0:
            print("Accuracy:", drafter.test_accuracy())
    
        try:
            r = req.result()
        except (requests.ConnectionError, requests.Timeout, socket.timeout):
            #print("Unexpected error:", sys.exc_info()[0])
            errors += 1
            time.sleep(sleep_time/1000)
            req = next_matches()
            continue
        
        if (r.status_code != 200):
            errors += 1
            time.sleep(sleep_time/1000)
            req = next_matches()
            continue
        if 'matches' in r.json()['result']:
            matches = r.json()['result']['matches']
        else:
            time.sleep(sleep_time/1000)
            req = next_matches()
            continue

        if len(matches) == 0:
            time.sleep(sleep_time/1000)
            req = next_matches()
            continue
            
        next_id = matches[-1]['match_seq_num'] + 1 #+ ((0 * len(matches)) // 100)
        req = next_matches()
        
        matches_bulk = list()

        for match in matches:
            if match['lobby_type'] == 7 or match['lobby_type'] == 2 or match['lobby_type'] == 5:
                #add match to list
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

                start_time = dt.datetime.fromtimestamp(matches[-1]['start_time'])
                if not latest_start_time or start_time > latest_start_time:
                    latest_start_time = start_time

        if len(matches_bulk) > 0:
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
                batch_ys[i2][not m['radiant_win']] = 1
                i2+=1
            drafter.train(batch_xs, batch_ys)
            
        #time.sleep(sleep_time/1000)

    except BaseException as e:
        drafter.save()
        db.close()
        sess.close()
        print("Errors: ", errors)
        print("Saved")
        raise e
    
drafter.save()
db.close()
sess.close()
print("Errors: ", errors)
print("Saved")
