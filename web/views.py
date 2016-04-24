from flask import render_template, request, send_from_directory
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
    mmr = int(request.args['mmr']) if 'mmr' in request.args else 4000
    return json.dumps(drafter.queryDraft(r_heroes, d_heroes, mmr))
    
@app.route('/robots.txt')
@app.route('/sitemap.txt')
def static_from_root():
    return send_from_directory(app.static_folder, request.path[1:])