from flask import Flask
import threading

app = Flask(__name__)
import drafterANN
drafter = drafterANN.DotoAnn()
from web import views

refresh_time = 60*60 # 1 hour
def refresh():
    threading.Timer(refresh_time, refresh).start()
    drafter.reload()
    print("Reloaded weights")

threading.Timer(refresh_time, refresh).start()