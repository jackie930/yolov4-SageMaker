# -*- coding: utf-8 -*-
import sys
import json
import boto3
import os
import warnings

warnings.filterwarnings("ignore",category=FutureWarning)

sys.path.append('/opt/program/textrank4zh')

import sys
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass


with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    # from autogluon import ImageClassification as task

import flask
import cv2 as cv

# The flask app for serving predictions
app = flask.Flask(__name__)

s3_client = boto3.client('s3')

def yolo_infer(weight,names,cfg,pic):
    #TODO: define endpoint output
    frame = cv.imread(pic)
    print ("<<<<pic shape:", frame.shape)

    net = cv.dnn_DetectionModel(cfg,weight)
    net.setInputSize(608, 608)
    net.setInputScale(1.0 / 255)
    net.setInputSwapRB(True)
    with open(names, 'rt') as f:
        names = f.read().rstrip('\n').split('\n')

    classes, confidences, boxes = net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)
    print ("<<<<done")
    return classes,confidences,boxes


@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    # health = ScoringService.get_model() is not None  # You can insert a health check here
    health = 1

    status = 200 if health else 404
    print("===================== PING ===================")
    return flask.Response(response="{'status': 'Healthy'}\n", status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def invocations():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    print("================ INVOCATIONS =================")

    #parse json in request
    print ("<<<< flask.request.content_type", flask.request.content_type)

    data = flask.request.data.decode('utf-8')
    data = json.loads(data)

    bucket = data['bucket']
    image_uri = data['image_uri']

    download_file_name = image_uri.split('/')[-1]
    print ("<<<<download_file_name ", download_file_name)

    s3_client.download_file(bucket, image_uri, download_file_name)
    #local test
    #download_file_name='./dog.jpg'
    print('Download finished!')
    # inference and send result to RDS and SQS

    print('Start to inference:')

    #LOAD MODEL
    weight = './yolov4.weights'
    names = './coco.names'
    cfg = './yolov4.cfg'

    #make sure the model parameters exist
    for i in [weight,names,cfg]:
        if os.path.exists(i):
            print ("<<<<pretrained model exists for :", i)
        else:
            print ("<<< make sure the model parameters exist for: ", i)
            break

    #make inference
    classes, confidences, boxes = yolo_infer(weight,names,cfg, download_file_name)
    #print("image_path:{},label:{}".format(download_file_name, label))
    print ("Done inference! ")
    inference_result = {
        'classes':classes.tolist(),
        'confidences':confidences.tolist(),
        'boxes':boxes.tolist()
    }
    _payload = json.dumps(inference_result,ensure_ascii=False)


    return flask.Response(response=_payload, status=200, mimetype='application/json')