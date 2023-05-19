import argparse
import io

import torch
from flask import Flask, request
from PIL import Image
import datetime

# import detect

import config
import pymysql

import json

import firebase_admin
from firebase_admin import credentials

from yolov5.detect import run

cred = credentials.Certificate("./serviceAccountKey.json")
firebase_admin.initialize_app(cred)

app = Flask(__name__)
models = {}
conn = pymysql.connect(
    host = config.db['host'],
    port = config.db['port'],
    user = config.db['user'],
    password = config.db['password'],
    db = config.db['database']
)


# API - Search
@app.route("/model/search/<model>", methods=['POST'])
def predict(model):
    if request.method != 'POST':
        return

    if request.files.get('image'):
        im_file = request.files['image']
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))
        print("image ok")

        if model in models:
            results = models[model](im, size=320)  # reduce size=320 for faster inference
            json_data = {}

            if (len(results.pandas().xyxy[0]) == 0):
                json_data["species"] = "발견된 해충이 없음"
                json_data["description"] = ""

            else:
                r = json.loads(results.pandas().xyxy[0]["name"].to_json(orient="records"))

                if(r[0] == 'Cockroach'):
                    sql = "select * from bug where id = %s"
                    cursor = conn.cursor()
                    cursor.execute(sql, 1)
                    info = cursor.fetchone()
                    species = info[1]
                    description = info[2]

                    json_data["species"] = species
                    json_data["description"] = description

                    conn.commit()
            
        return json_data
    
    return "fail"




def save2(user_id, bug_id, result_url):

    cursor = conn.cursor()
    cursor.execute("INSERT INTO scenario (user_id, bug_id, image) VALUES(%s, %s, %s)", (user_id, bug_id, result_url))
    conn.commit()
        
    return "success"
    


@app.route("/model/video", methods=['POST'])
def analyze2():
    if request.method != 'POST':
        return
    
    # params = request.get_json()
    # user_id = params["user_id"]
    # video_url = params["viedo_url"]

    run(source='https://firebasestorage.googleapis.com/v0/b/debugging-eb903.appspot.com/o/Project001.mp4?alt=media&token=2f903157-f382-4f5c-ac3a-0bd986c08df5', weights='./yolov5/runs/train/model_v3/weights/best.pt')

    # model(video_url) = { bug_id, res_url }
    # save2(user_id, bug_id, result_url)
    


    return "hello"


# API - Save scenario
@app.route("/scenario", methods=['POST'])
def save():
    if request.method != 'POST':
        return
    
    params = request.get_json()
    cursor = conn.cursor()

    if params:
        userId = params['userId']
        bugId = params['bugId']
        image = params['image']
        createdAt =  datetime.datetime.now()
        
        cursor.execute("INSERT INTO scenario (user_id, bug_id, image, created_at) VALUES(%s, %s, %s, %s)", (userId, bugId, image, createdAt))
        conn.commit()
        
        return "success"
    
    return



# API -  Create Report
@app.route("/report/<model>", methods=['POST'])
def analyze(model):
    if request.method != 'POST':
        return
    
    params = request.get_json()
    
    if params:
        video_url = params['url']
        
        # if model in models:
        results = models[model](size = 1920, source = video_url)
        print(results.to_json(orient='records'))
        return "hello"
    
    return


def create_app():
    app.config.from_pyfile("config.py")
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flask API exposing YOLOv5 model')
    parser.add_argument('--port', default=5000, type=int, help='port number')
    parser.add_argument('--model', nargs='+', default=['yolov5s'], help='model(s) to run, i.e. --model yolov5n yolov5s')
    opt = parser.parse_args()

    for m in opt.model:
        models[m] = torch.hub.load('ultralytics/yolov5', 'custom', './yolov5/runs/train/model_v3/weights/best.pt', force_reload=True, skip_validation=True)

    create_app().run('0.0.0.0', port=5000, debug=False)
    #app.run(host='0.0.0.0', port=opt.port)  # debug=True causes Restarting with stat