# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import os

import sys

import json
from boto3.session import Session
import multiprocessing as mul

from elasticsearch import Elasticsearch, RequestsHttpConnection
from datetime import datetime
from bbox import main

region_name = os.getenv("region_name")
endpoint_name = os.getenv("endpoint_name")
output_s3_bucket = os.getenv("output_s3_bucket")
output_s3_prefix = os.getenv("output_s3_prefix")
elastic_search_host = os.getenv("es_host")
elastic_search_port = os.getenv("es_port")
elastic_search_protocol = os.getenv("es_protocol")
batch_id = os.getenv("batch_id")
job_id = os.getenv("job_id")
elastic_search_index = os.getenv('es_index')
import pandas as pd
import sys
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

# Hack to print to stderr so it appears in CloudWatch.
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


eprint(
    ">>>> start the job with param - outside es_host-{}, batch_id - {}, region_name-{}, endpoint_name-{},\n \
     output_s3_bucket-{}, output_s3_prefix-{}, es_port-{}, es_protocol-{}, job_id-{}, ".format(
        elastic_search_host, batch_id, region_name, endpoint_name, output_s3_bucket, output_s3_prefix,
        elastic_search_port, elastic_search_protocol, job_id))


def text_summary_main(input_s3_path_list: list, endpoint_name, output_s3_bucket, output_s3_prefix, region_name):
    """
    function: save json_files back to s3 (key: file name value: tag label)
    :param input_s3_path_list: 读入s3文件路径list
    :param endpoint_path: 保存名称
    """
    session = Session(
        region_name=region_name
    )
    s3 = session.client("s3")
    es = __connect_ES()
    eprint("start!")
    for x in input_s3_path_list:
        s3_path= "s3://{}/{}".format(x["_source"]["bucket"], x["_source"]["file_key"])
        result = {}
        bucket = s3_path.split("/")[2]
        key = "/".join(s3_path.split("/")[3:])
        file_name = s3_path.split("/")[-1]
        # try:
        # s3.Bucket(bucket).download_file(key, file_name)
        eprint(file_name, key, bucket)
        s3.download_file(Filename=file_name, Key=key, Bucket=bucket)

        try:

            # read image
            eprint("process", s3_path)
            key_list = ["name_of_business", "branch_name", "address",
                        "nature_of_business", 'status', 'date_of_comm',
                        'data_of_expiry', 'certificate_no', 'fee_and_levy']
            res_pos,Res = main(file_name,model='relat')
            img_list = os.listdir('./res')
            output_s3_subimg_prefix = 'sub_images'
            infer_ls = []
            for i in img_list:
                if i.split('_')[0]=='box':
                    try:
                        upload_key=output_s3_subimg_prefix+'/'+i
                        s3.upload_file(Filename=os.path.join('./res',i), Key=upload_key, Bucket=output_s3_bucket)
                        print("uploaded to s3://{}/{}".format(output_s3_bucket, upload_key))
                        print ("<<updload key: ", upload_key)
                        infer_ls.append(upload_key)
                    except:
                        continue

            #infer on sub-images
            pool = mul.Pool(10)
            res = pool.map(infer, infer_ls)
            print ("<<<<<<<<<<<<<<<")
            print ("<<<< res", res)

            label = post_process(res,res_pos)

        except:
            label = "NA"

        print ("<<<< label", label)
        result[s3_path] = label

        # save json file
        file_name_fre = ''.join(file_name.split('.')[:-1])
        json_file = file_name_fre+".json"
        print ("<<<json file: ", json_file)
        with open(json_file, "w") as fw:  # 建议改为.split('.')
            json.dump(result, fw, ensure_ascii=False)
            eprint("write json file success!")

        # output to s3
        print ("<<<< output_s3_prefix: ",output_s3_prefix)
        upload_key=output_s3_prefix+'/'+json_file
        s3.upload_file(Filename=json_file, Key=upload_key, Bucket=output_s3_bucket)
        print("uploaded to s3://{}/{}".format(output_s3_bucket, upload_key))

        # update elasticsearch
        doc_id = x["_id"]
        update_status_by_id(es, doc_id, status="COMPLETED", output=str(label))

        # delete file locally
        delete_file(json_file)
        delete_file(file_name)

def infer(input_image):
    #from boto3.session import Session
    import json

    image_uri = input_image
    test_data = {
        'bucket' : output_s3_bucket,
        'image_uri' : image_uri,
        'content_type': "application/json",
    }
    payload = json.dumps(test_data)
    session = Session(region_name=region_name)

    runtime = session.client("runtime.sagemaker")
    response = runtime.invoke_endpoint(
        EndpointName='esd-crnn',
        ContentType="application/json",
        Body=payload)

    result = json.loads(response["Body"].read().decode(encoding='utf-8').encode(encoding='utf-8'))
    print ("<<< predict result: ",result)
    res = input_image+': ' + result['result']
    return res

def delete_file(file):
    """
    delete file
    :param file:
    :return:
    """
    if os.path.isfile(file):
        try:
            os.remove(file)
        except:
            pass

def convert_to_key(x):
    a = x.split('/')[1]
    a = ('_'.join(a.split('_')[1:]))
    return a[:-5]

def convert_to_key_index(x):
    a = x.split('/')[1]
    a = ('_'.join(a.split('_')[1:]))
    return a[-5]

def post_process(x,Res):
    keys = [i.split(':')[0] for i in x]
    vals = [i.split(':')[1] for i in x]
    key_map = [convert_to_key(x) for x in keys]
    print ("<<<<< key map: ",key_map)
    key_index_map = [convert_to_key_index(x) for x in keys]
    #dataframe for result from prediction
    df = pd.DataFrame({'x1':keys,'x2':vals,'x3':key_map,'x4':key_index_map})

    print ("head of df: ",df.head())
    print ("<<<<<<< RES", Res)
    keys_res = list(Res.keys())
    pos_res = list(Res.values())
    for i in range(len(pos_res)):
        for j in range(len(pos_res[i])):
            pos_res[i][j]=pos_res[i][j].tolist()

    df2 = pd.DataFrame({'x3':keys_res,'pos':pos_res})
    print ("head of df2", df2.head())
    print ("df2 keys: ", df2.x3)
    print ("df keys: ", df.x3)

    df_res = pd.merge(df,df2,on='x3')
    print ("df_res: ", df_res.head())
    df_res = df_res.sort_values(by=["x3","x4"])
    label = {'keys':df_res["x3"].tolist(),
             "vals":df_res["x2"].tolist(),
             "pos":df_res["pos"].tolist()}

    return label

def __connect_ES() -> Elasticsearch:
    eprint('Connecting to the ES Endpoint {}:{}'.format(elastic_search_host, elastic_search_port))
    es = None
    print("Using protocol: ", elastic_search_protocol)
    try:
        if elastic_search_protocol == "http":
            eprint (">>> Using http")
            es = Elasticsearch(
                hosts=[{'host': elastic_search_host, 'port': elastic_search_port}],
                # http_auth=awsauth,
                connection_class=RequestsHttpConnection)
        else:
            eprint (">>> Using https")
            es = Elasticsearch(
                hosts=[{'host': elastic_search_host, 'port': elastic_search_port}],
                use_ssl=True
            )

    except Exception as E:
        eprint("Unable to connect to {0}")
        eprint(E)
        exit(3)

    return es


def __search_for_file_list(job_id: str, batch_id: str, status='NOT_STARTED') -> list:
    """
    Fot bot to get their file to process.
    :param job_id: One customer request for bots to process on certain folder is one *job*, and thus have a job_id.
    :param batch_id: Files in one job will spited based on the number of bots. Bot id and batch id is the same.
    :param status: one of "NOT_STARTED, COMPLETED, PROCESSING"
    :return:
    """
    es = __connect_ES()
    query = {  # TODO Would speed up by reduce the result fields
        "size": 10000,
        "query": {
            "bool": {
                "must": [
                    {"match": {"job_id": job_id}},
                    {"match": {"batch_id": batch_id}},
                    {"match": {"status": status}}
                ]
            }
        }
    }
    eprint(">>> going to query: {}".format(query))

    resp = es.search(
        index=elastic_search_index,
        body=query,
        scroll='9s'
    )

    # keep track of pass scroll _id
    old_scroll_id = resp['_scroll_id']
    all_hits = resp['hits']['hits']
    # use a 'while' iterator to loop over document 'hits'
    while len(resp['hits']['hits']):

        # make a request using the Scroll API
        resp = es.scroll(
            scroll_id=old_scroll_id,
            scroll='2s'  # length of time to keep search context
        )

        # check if there's a new scroll ID
        if old_scroll_id != resp['_scroll_id']:
            eprint("NEW SCROLL ID:", resp['_scroll_id'])

        # keep track of pass scroll _id
        old_scroll_id = resp['_scroll_id']

        # eprint the response results
        eprint("\nresponse for index: " + elastic_search_index)
        eprint("_scroll_id:", resp['_scroll_id'])
        eprint('response["hits"]["total"]["value"]:{}'.format(resp["hits"]["total"]))

        # iterate over the document hits for each 'scroll'
        all_hits.extend(resp['hits']['hits'])
    eprint("DOC COUNT:", len(all_hits))

    return all_hits


def update_status_by_id(es, doc_id, status="COMPLETED", output=""):
    resp = es.update(
        index=elastic_search_index,
        id=doc_id,
        body={
            "doc":
                {"status": status,
                 "output": output,
                 "complete_date": datetime.utcnow()
                 }
        },
        doc_type="_doc"
    )
    return resp

if __name__ == "__main__":
    eprint(">>> Start execution.")

    file_list = __search_for_file_list(job_id=job_id, batch_id=batch_id)
    text_summary_main(file_list, endpoint_name, output_s3_bucket, output_s3_prefix,
                      region_name)
    eprint("<<< Exit.")