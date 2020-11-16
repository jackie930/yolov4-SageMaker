ARG REGISTRY_URI
#FROM ${REGISTRY_URI}/mxnet-inference:1.6.0-cpu-py3
FROM ${REGISTRY_URI}/pytorch-inference:1.5.0-cpu-py36-ubuntu16.04

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

ENV LANG=en_US.utf8
ENV LANG=C.UTF-8

COPY . /tmp
WORKDIR /tmp
RUN pip install boto3 -i https://opentuna.cn/pypi/web/simple
RUN pip install glog -i https://opentuna.cn/pypi/web/simple
RUN pip install elasticsearch==7.8.0 -i https://opentuna.cn/pypi/web/simple
RUN pip install requests -i https://opentuna.cn/pypi/web/simple
RUN pip install opencv-python==4.4.0.40 -i https://opentuna.cn/pypi/web/simple
RUN pip install scipy fitz PyMuPDF imutils==0.5.3 -i https://opentuna.cn/pypi/web/simple
#RUN pip install torch -i https://opentuna.cn/pypi/web/simple
#CMD ["python","-u","/tmp/task.py"]

ENTRYPOINT ["python", "/tmp/task.py"]