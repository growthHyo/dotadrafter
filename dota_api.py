import requests
from requests_futures.sessions import FuturesSession
import socket
import time
import sys
from requests_toolbelt.adapters import source

class Dota_API():
    api_keys = ['FE70CE9FC0D6D99279498CE852587F59','2FEC67172AAC0C393EC209A225A7E51E']
    api_key_num = 1
    api_key = api_keys[api_key_num]
    
    ips = ['162.213.199.143', '162.213.199.31']
    ip_num = 0
    
    data_source = 4
    
    headers = {'User-Agent': 'Script by Grue'}

    errors = 0

    session = FuturesSession()
    session.mount('http://', source.SourceAddressAdapter(ips[ip_num]))

    def matches_get(self, req_type=1, n_id='', **kwargs):
        if (req_type < 4):
            url = 'https://api.steampowered.com/IDOTA2Match_570/'
            url += 'GetMatchHistoryBySequenceNum' if req_type==1 else 'GetMatchHistory'
            url += '/V001/?key=' + self.api_key + '&min_players=10&'
            if (req_type==1):
                url += 'start_at_match_seq_num'
            elif (req_type==2):
                url += 'start_at_match_id'
            elif (req_type==3):
                url += 'account_id'
            url += '=' + str(n_id)
            if req_type != 1:
                url += '&skill=3'
        elif req_type == 4:
            url = 'http://www.dotabuff.com/matches/' + str(n_id)
        elif req_type == 5:
            url = 'http://dotamax.com/match/detail/' + str(n_id)
        elif req_type == 6:
            url = 'http://api.opendota.com/api/matches/' + str(n_id)
        return dict(req=self.session.get(url, timeout=7, headers=self.headers), req_type=req_type, n_id=n_id, url=url, ip_num=self.ip_num)

    def matches_result(self, request):
        req = request['req']
        try:
            res = req.result()
        except (requests.ConnectionError, requests.Timeout, socket.timeout) as e:
            return self.retry_request(request)
        if (res.status_code != 200):
            if (res.status_code == 404 and request['req_type'] == 6):
                #not found
                return None
            self.session = FuturesSession()
            # if last IP cycle through data sources
            if self.ip_num == len(self.ips) - 1:
                if request['req_type'] == 4 or request['req_type'] == 6:
                    request['req_type'] = (4 if (request['req_type'] == 6) else 6)
                    self.data_source = request['req_type']
            self.ip_num = (request['ip_num']+1) % len(self.ips)
            self.session.mount('http://', source.SourceAddressAdapter(self.ips[self.ip_num]))
            return self.retry_request(request, sleep=1)
        if request['req_type']==4:
            return self.parse_skill(res)
        if request['req_type']==5:
            return self.parse_dota_max(res)
        if request['req_type']==6:
            #switch IPs and wait 0.5 seconds so that it is 1 request per second per IP
            time.sleep(1/len(self.ips))
            self.ip_num = (request['ip_num']+1) % len(self.ips)
            self.session.mount('http://', source.SourceAddressAdapter(self.ips[self.ip_num]))
            return self.parse_opendota_skill(res)
        try:
            matches = res.json()['result']['matches']
        except:
            if (request['req_type']==3):
                return []
            return self.retry_request(request)
        if len(matches) == 0:
            return self.retry_request(request)
        return matches


    def retry_request(self, request, sleep=7):
        #print(request)
        self.errors += 1
        time.sleep(sleep)
        return self.matches_result(self.matches_get(**request))
    
    def parse_skill(self, response):
        html = response.text
        end_index = html.find(' Skill</dd>')
        if end_index > -1:
            html = html[:end_index]
        else:
            return None
        start_index = html.rfind('<dd>')
        if start_index > -1:
            html = html[start_index+4:]
        else:
            return None
        return html
    
    def parse_dota_max(self, response):
        html = response.text
        html_split = html.split('<td><font style="color: #f0a868;">')
        if len(html_split) > 1:
            html = html_split[1]
        else:
            return None
        html_split = html.split('</font></td>')
        if len(html_split) > 1:
            html = html_split[0]
        else:
            return None
        return html
    
    def parse_opendota_skill(self, response):
        m = response.json()
        if 'skill' not in m:
            return None
        if m['skill'] == 3:
            return 'Very High'
        return m['skill']
    