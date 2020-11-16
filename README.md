# yolov4-SageMaker
Deploy YOLO-V4 on Amazon SageMaker


<a name="YOLOv4"></a>

## Features

- [x] **Use opencv-4.4.0.40 to deploy yolov4**
- [x] **Use flask+docker to deploy the rest-api locally**
- [x] **Use Amazon SageMaker to deploy the endpoint**
- [x] **Suopport video input loccally**
- [ ] **video queue infer**
- [ ] **deploy yolov4-image-bot on spot bot**
- [ ] **deploy yolov4-video-bot on spot bot**


## Quick Start

---------------

first download yolov4 public model and put under pretrained_model folder

```shell script
mkdir source/yolov4_endpoint/pretrained_model
git clone https://github.com/AlexeyAB/darknet.git
cp ./darknet/cfg/yolov4.cfg ./source/yolov4_endpoint/pretrained_model/
cp ./darknet/cfg/coco.names ./source/yolov4_endpoint/pretrained_model/
cd source/yolov4_endpoint/pretrained_model
#download weight
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
```

The file stucture will look like below
```shell script
|-- source
    |-- yolov4_endpoint
        |-- pretrained_model
            |-- yolov4.cfg
            |-- yolov4.weights
            |-- coco.names
        |-- build_and_push.sh
        ...
```
   
# build the environment and test locally

server
~~~~shell script
sh build_and_push.sh
docker run -v -d -p 8080:8080 yolov4
~~~~

client
~~~~shell script
import requests
import json

url='http://localhost:8080/invocations'

bucket = 'spot-bot-asset'
image_uri = 'end/dog.jpg'
test_data = {
    'bucket' : bucket,
    'image_uri' : image_uri,
    'content_type': "application/json",
}
payload = json.dumps(test_data)


r = requests.post(url,data=payload)

#show result
print (r.text)
~~~~

result
~~~~
{"classes": [[1], [7], [16], [58]], "confidences": [[0.9237534403800964], [0.9179147481918335], [0.979065477848053], [0.33346137404441833]], "boxes": [[114, 127, 458, 298], [464, 77, 220, 93], [128, 225, 184, 316], [681, 109, 36, 45]]}
CPU times: user 5.48 ms, sys: 174 µs, total: 5.65 ms
Wall time: 1.66 s
~~~~

# build endpoint
~~~~
cd yolov4_endpoint
python create_endpoint.py
~~~~

# use endpoint
~~~~ python
def infer(input_image):
    from boto3.session import Session
    import json

    bucket = 'predictive-solution'
    image_uri = input_image
    test_data = {
        'bucket' : bucket,
        'image_uri' : image_uri,
        'content_type': "application/json",
    }
    payload = json.dumps(test_data)


    session = Session()

    runtime = session.client("runtime.sagemaker")
    response = runtime.invoke_endpoint(
        EndpointName='yolov4',
        ContentType="application/json",
        Body=payload)

    result = json.loads(response["Body"].read())
    print (result)
    
infer('yolotest/dog.jpg')
~~~~

~~~~text
{'classes': [[1], [7], [16], [58]], 'confidences': [[0.9237533807754517], [0.917914867401123], [0.979065477848053], [0.3334614336490631]], 'boxes': [[114, 127, 458, 298], [464, 77, 220, 93], [128, 225, 184, 316], [681, 109, 36, 45]]}
CPU times: user 116 ms, sys: 8.93 ms, total: 125 ms
Wall time: 6.04 s
~~~~

## local infer on image
todo
## local infer on video
~~~~
python yolov4_endpoint/infer_video.py 
~~~~