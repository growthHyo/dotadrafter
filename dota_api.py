import requests
from requests_futures.sessions import FuturesSession
import socket
import time

class Dota_API():
    api_keys = ['FE70CE9FC0D6D99279498CE852587F59','2FEC67172AAC0C393EC209A225A7E51E']
    api_key_num = 1
    api_key = api_keys[api_key_num]
    sleep_time = 7

    errors = 0

    session = FuturesSession()

    def matches_get(self, seq_num=True, n_id='', **kwargs):
        url = 'https://api.steampowered.com/IDOTA2Match_570/'
        url += 'GetMatchHistoryBySequenceNum' if seq_num else 'GetMatchHistory'
        url += '/V001/?key=' + self.api_key + '&min_players=10&'
        url += 'start_at_match_seq_num' if seq_num else 'start_at_match_id'
        url += '=' + str(n_id)
        if not seq_num:
            url += '&skill=3&game_mode=22'
        return dict(req=self.session.get(url, timeout=4), seq_num=seq_num, n_id=n_id, url=url)

    def matches_result(self, request):
        req = request['req']
        try:
            res = req.result()
        except (requests.ConnectionError, requests.Timeout, socket.timeout):
            #print("Unexpected error:", sys.exc_info()[0])
            return self.retry_request(request)
        if (res.status_code != 200):
            return self.retry_request(request)
        if not 'matches' in res.json()['result']:
            return self.retry_request(request)
        matches = res.json()['result']['matches']
        if len(matches) == 0:
            return self.retry_request(request)
        return matches


    def retry_request(self, request):
        self.errors += 1
        #self.api_key_num = abs(self.api_key_num - 1)
        #self.api_key = self.api_keys[self.api_key_num]
        time.sleep(self.sleep_time)
        return self.matches_result(self.matches_get(**request))

