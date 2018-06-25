FROM mxnet/python
# python bottle
RUN pip install bottle --upgrade
RUN mkdir -p /workspace/image-predict
COPY image-predict/server.py /workspace/image-predict/
COPY image-predict/predictor /workspace/image-predict/predictor
COPY image-predict/model /workspace/image-predict/model
WORKDIR /workspace/image-predict
CMD ["python", "server.py"]