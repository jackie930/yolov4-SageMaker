# -*- coding: utf-8 -*-
import sys
import os
import argparse
import logging
import warnings
import io
import json
import boto3

import warnings
import numpy as np
import crnn
import torch
from PIL import Image
import itertools
import cv2

warnings.filterwarnings("ignore",category=FutureWarning)

sys.path.append('/opt/program/textrank4zh')

import sys
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

import codecs

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
    net = cv.dnn_DetectionModel(cfg,weight)
    net.setInputSize(608, 608)
    net.setInputScale(1.0 / 255)
    net.setInputSwapRB(True)
    frame = cv.imread(pic)
    with open(names, 'rt') as f:
        names = f.read().rstrip('\n').split('\n')

    classes, confidences, boxes = net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)
    for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
        label = '%.2f' % confidence
        label = '%s: %s' % (names[classId], label)
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        left, top, width, height = box
        top = max(top, labelSize[1])
        cv.rectangle(frame, box, color=(0, 255, 0), thickness=3)
        cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv.imwrite('./res.jpg', frame)
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
    #download_file_name = './test.jpg'
    #s3_client.download_file(bucket, image_uri, download_file_name)
    #local test
    download_file_name='./dog.jpg'
    print('Download finished!')
    # inference and send result to RDS and SQS

    print('Start to inference:')

    #LOAD MODEL
    weight = './pretrained_model/yolov4.weights'
    names = './pretrained_model/coco.names'
    cfg = './pretrained_model/yolov4.cfg'

    #make inference
    classes, confidences, boxes = yolo_infer(weight,names,cfg, download_file_name)
    #print("image_path:{},label:{}".format(download_file_name, label))
    print ("Done inference! ")
    inference_result = {
        'classes':classes,
        'confidences':confidences,
        'boxes':boxes
    }
    _payload = json.dumps(inference_result,ensure_ascii=False)


    return flask.Response(response=_payload, status=200, mimetype='application/json')