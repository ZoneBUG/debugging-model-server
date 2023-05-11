import argparse
import io

import torch
from flask import Flask, request
from PIL import Image

import config
from db_connect import db

from flask_jsonpify import jsonpify 
import json

import firebase_admin
from firebase_admin import credentials

cred = credentials.Certificate("./serviceAccountKey.json")
firebase_admin.initialize_app(cred)

app = Flask(__name__)
models = {}

@app.route("/model/search/<model>", methods=['POST'])
def predict(model):
    if request.method != 'POST':
        return

    if request.files.get('image'):
        im_file = request.files['image']
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))

        if model in models:
            results = models[model](im, size=320)  # reduce size=320 for faster inference
            json_data = {}

            if (len(results.pandas().xyxy[0]) == 0):
                json_data["type"] = "발견된 해충이 없음"

            else:
                r = json.loads(results.pandas().xyxy[0]["name"].to_json(orient="records"))
                json_data["type"] = r[0]
            
        return json_data


@app.route("/report/<model>", methods=['POST'])
def analyze(model):
    if request.method != 'POST':
        return
    
    params = request.get_json()
    if params:
        video_url = params['url']
        created_at = params['created_at']
        
        if model in models:
            results = models[model](video_url, size = 1920)
            print(results.pandas().xyxy[0][0].to_json(orient='records'))
            return "hello"


def create_app():
    app.config.from_object(config)
    db.init_app(app)
    
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flask API exposing YOLOv5 model')
    parser.add_argument('--port', default=5000, type=int, help='port number')
    parser.add_argument('--model', nargs='+', default=['yolov5s'], help='model(s) to run, i.e. --model yolov5n yolov5s')
    opt = parser.parse_args()

    for m in opt.model:
        models[m] = torch.hub.load('ultralytics/yolov5', 'custom', './yolov5/best.pt', force_reload=True, skip_validation=True)

    create_app().run()
    #app.run(host='0.0.0.0', port=opt.port)  # debug=True causes Restarting with stat