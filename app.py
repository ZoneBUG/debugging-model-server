import argparse
import io

import urllib.request
import os
import datetime


from flask import Flask, request

from PIL import Image
import torch

import config
import pymysql

import json

import firebase_admin
from firebase_admin import credentials, storage

from yolov5.detect import run

cred = credentials.Certificate("./serviceAccountKey.json")
firebase_admin.initialize_app(cred, {'storageBucket': config.fb_storage_bucket})
bucket = storage.bucket()

app = Flask(__name__)
models = {}
conn = pymysql.connect(
    host = config.db['host'],
    port = config.db['port'],
    user = config.db['user'],
    password = config.db['password'],
    db = config.db['database']
)

bug_list = {
	"cockroach": 1,
	"scutigeridae": 2,
	"centipede": 4,
	"diplopoda": 5,
	"silverfish": 6,
	"dermaptera": 7
}


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
                
                if(r[0] in bug_list):
                    bug_id = bug_list[r[0]]
                    sql = "select * from bug where id = %s"
                    cursor = conn.cursor()
                    cursor.execute(sql, bug_id)
                    info = cursor.fetchone()
                    species = info[1]
                    description = info[2]

                    json_data["species"] = species
                    json_data["description"] = description

                    conn.commit()
            
        return json_data
    
    return "fail"



# API - Create & Save Scenario for Report
@app.route("/model/video", methods=['POST'])
def analyze2():
    if request.method != 'POST':
        return
    
    created_at =  datetime.datetime.now()
    params = request.get_json()
    user_id = params["user_id"]
    video_url = params["url"]

    video_path = os.path.join('./videos', 'input_video.mp4')
    urllib.request.urlretrieve(video_url, video_path)

    try :
        result = run(weights='./yolov5/runs/train/model_v3/weights/best.pt', source=video_path)
        img_dir_path = result[0]
        object_name = result[1]

        # 영상 분석 결과, 발견된 해충 없는 경우
        if(object_name not in bug_list): 
            return "fail"
        
        bug_id = bug_list[object_name]            

        # Firebase Storage에 이미지 업로드
        firebase_img_url = uploadImgToFirebase(img_dir_path, user_id)

        # RDS에 Scenario INSERT
        return saveScenario(user_id, bug_id, firebase_img_url, created_at)

    except:
        return "fail"


# Firebase Storage에 결과 이미지 업로드 후 url 경로 리턴
def uploadImgToFirebase(img_dir_path, user_id):
    local_file_path = "./yolov5/runs/detect/" + img_dir_path + "/input_video.png"
    remote_file_path = "Report/" + str(user_id) + "/" + img_dir_path
    blob = bucket.blob(remote_file_path)

    blob.upload_from_filename(filename=local_file_path, content_type='image/png')   # Uplad Image to Firebase Strage
    blob.make_public()  # Get Public URL from blob

    return blob.public_url


# RDS에 Scenario 저장
def saveScenario(user_id, bug_id, img_url, created_at):
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO scenario (user_id, bug_id, image, created_at) VALUES(%s, %s, %s, %s)", 
        (user_id, bug_id, img_url, created_at)
    )
    conn.commit()
        
    return "success"



# Config 설정
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
