from flask import render_template, request
from web import app, drafter
import json

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', heroes=list(drafter.heroes.values()) + list(drafter.hero_translations))
    
@app.route('/calc')
def calc():
    r_heroes = request.args['r_hero_input'].split(',')
    d_heroes = request.args['d_hero_input'].split(',')
    return json.dumps(drafter.queryDraft(r_heroes, d_heroes))
    