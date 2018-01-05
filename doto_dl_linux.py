import datetime as dt
import time
from drafterANN import *
from dota_api import Dota_API
import sys

sys.setrecursionlimit(2000)

drafter = DotoAnn()
api = Dota_API()

next_id = '2238020584'
latest_start_time = None

for m in Match.select().order_by(Match.seq_num.desc()).limit(1):
    next_id = m.seq_num + 1

i = 0
matches_parsed = 0
matches_trained = 0

req = api.matches_get(n_id=next_id)

matches_bulk = list()
train_batch = list()
batch_size = 128

try:
    while True:
        try:
            i += 1
            if i%300==1:
                print('Errors: ', api.errors)
                print('Matches: ', matches_parsed)
                print('Trained: ', matches_trained)
                api.errors = 0
                matches_parsed = 0
                matches_trained = 0
                print(time.strftime("%H:%M:%S") + ": " + str(latest_start_time))

                inp = np.zeros((1, n_input), np.int)
                inp[0][2] = 1
                print("Axe: " + str(drafter.run(inp)))

                inp = np.zeros((1, n_input), np.int)
                inp[0][2 + max_heroes] = 1
                print("vs Axe: " + str(drafter.run(inp)))

                print("")
                drafter.save()

            if i%1500==1:
                print("Accuracy:", drafter.test_accuracy())

            matches = api.matches_result(req)

            next_id = matches[-1]['match_seq_num'] + 1
            if latest_start_time and dt.datetime.now() - latest_start_time > dt.timedelta(hours=5):
                next_id += 5000
            if latest_start_time and dt.datetime.now() - latest_start_time < dt.timedelta(hours=1):
                time.sleep(60)
            req = api.matches_get(n_id=next_id)
            start_time = dt.datetime.fromtimestamp(matches[-1]['start_time'])
            if not latest_start_time or start_time > latest_start_time:
                latest_start_time = start_time

            reqs_dict = dict()
            matches_dict = dict()

            for match in matches:
                if match['human_players'] < 10:
                    continue
                # 7 == Ranked, 2 == Tournament, 5 == Team match, 1 == Practice
                # https://wiki.teamfortress.com/wiki/WebAPI/GetMatchDetails
                if not (match['lobby_type'] == 7 or match['lobby_type'] == 2 or match['lobby_type'] == 5 or match['lobby_type'] == 1):
                    continue
                # 2 == Captains Mode
                #if not match['game_mode'] == 2:
                #    continue
                abandon = False
                for player in match['players']:
                    if player['leaver_status'] > 4:
                        abandon = True
                if abandon:
                    continue
                m_id = match['match_id']
                reqs_dict[m_id] = api.matches_get(api.data_source, m_id)
                matches_dict[m_id] = match
            for m_id, req2 in reqs_dict.items():
                skill = api.matches_result(req2)
                if (skill == 'Very High' or True):   
                    #add match to list
                    match = matches_dict[m_id]
                    r_heroes = []
                    d_heroes = []
                    # if captains mode segment the matches by pick order
                    if 'picks_bans' in match:
                        for pickban in match['picks_bans']:
                            if pickban['is_pick']:
                                m = dict(id=match['match_id'], seq_num=match['match_seq_num'], radiant_win=match['radiant_win'])
                                if pickban['team'] == 0:
                                    r_heroes.append(str(pickban['hero_id']))
                                else:
                                    d_heroes.append(str(pickban['hero_id']))
                                m['radiant_heroes'] = ",".join(r_heroes)
                                m['dire_heroes'] = ",".join(d_heroes)
                                train_batch.append(m)
                                if (len(r_heroes) + len(d_heroes) == 10):
                                    matches_bulk.append(m)
                    else:
                        m = dict(id=match['match_id'], seq_num=match['match_seq_num'], radiant_win=match['radiant_win'])
                        for player in match['players']:
                            if player['player_slot'] < 100:
                                r_heroes.append(str(player['hero_id']))
                            else:
                                d_heroes.append(str(player['hero_id']))
                        m['radiant_heroes'] = ",".join(r_heroes)
                        m['dire_heroes'] = ",".join(d_heroes)
                        matches_bulk.append(m)
                        train_batch.append(m)

                    #train on batch
                    if len(matches_bulk) >= batch_size:
                        #save matches to db
                        with db.atomic():
                            Match.insert_many(matches_bulk).execute()

                        #train the network
                        batch_xs = np.zeros((len(train_batch), n_input), np.int)
                        batch_ys = np.zeros((len(train_batch), 2), np.int)
                        i2=0
                        for m in train_batch:
                            for h in m['radiant_heroes'].split(","):
                                if len(h) > 0:
                                    batch_xs[i2][int(h)] = 1
                            for h in m['dire_heroes'].split(","):
                                if len(h) > 0:
                                    batch_xs[i2][int(h) + max_heroes] = 1
                            batch_ys[i2][0 if m['radiant_win'] else 1] = 1
                            i2+=1
                        drafter.train(batch_xs, batch_ys)
                        matches_parsed += len(matches_bulk)
                        matches_trained += len(train_batch)
                        matches_bulk = list()
                        train_batch = list()
        except Exception as e:
            #raise e
            print(e)
        finally:
            drafter.save()

except BaseException as e:
    raise e
finally:
    drafter.save()
    db.close()
    print("Errors: ", api.errors)
    print("Matches: ", matches_parsed)
    print("Saved")
