# -*- coding: utf-8 -*-
# @Time    : 11/13/20 6:46 PM
# @Author  : Jackie
# @File    : infer_video.py
# @Software: PyCharm

import cv2 as cv
import matplotlib.pyplot as plt
import time
import imutils

# todo: use queue to speed up https://github.com/AlexeyAB/darknet/blob/master/darknet_video.py

def yolo_infer(weight,cfg,frame):
    net = cv.dnn_DetectionModel(cfg,weight)
    net.setInputSize(608, 608)
    net.setInputScale(1.0 / 255)
    net.setInputSwapRB(True)
    #frame = cv.imread(pic)

    classes, confidences, boxes = net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)
    print ("<<<<done")
    return classes,confidences,boxes


def detect_objects(weight,names,cfg,video):

    # get video frames and pass to YOLO for output

    cap = cv.VideoCapture(video)
    writer = None
    # try to determine the total number of frames in the video file
    try:
        prop = cv.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv.CAP_PROP_FRAME_COUNT
        total = int(cap.get(prop))
        print("[INFO] {} total frames in video".format(total))
    # an error occurred while trying to determine the total
    # number of frames in the video file
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1

    i=0
    # initialize video stream, pointer to output video file and grabbing frame dimension

    while(cap.isOpened()):
        stime= time.time()
        ret, frame = cap.read()
        classes, confidences, boxes = yolo_infer(weight,cfg,frame)
        end = time.time()

        i = i+1
        print (i)

        if ret:
            for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                label = '%.2f' % confidence
                label = '%s: %s' % (names[classId], label)
                labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                left, top, width, height = box
                top = max(top, labelSize[1])
                cv.rectangle(frame, box, color=(0, 255, 0), thickness=3)
                cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv.FILLED)
                cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            if writer is None:
                # Initialize the video writer
                fourcc = cv.VideoWriter_fourcc(*"MP4V")
                writer = cv.VideoWriter('./res.mp4', fourcc, 30,
                                        (frame.shape[1], frame.shape[0]), True)
                # some information on processing single frame
                if total > 0:
                    elap = (end - stime)
                    print("[INFO] single frame took {:.4f} seconds".format(elap))
                    print("[INFO] estimated total time to finish: {:.4f}".format(
                        elap * total))


            writer.write(frame)

            print('FPS {:1f}'.format(1/(time.time() -stime)))
            if cv.waitKey(1)  & 0xFF == ord('q'):
                break
            if i>30:
                break
        else:
            break

    print ("<<<<clean up")
    writer.release()
    cap.release()
