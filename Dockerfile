FROM amazon/aws-lambda-python:3.8

COPY app.py ${LAMBDA_TASK_ROOT}
ADD checkpoint ./checkpoint
ADD models ./models

RUN yum install -y mesa-libGL

COPY requirements.txt  .
COPY build_custom_model.py  .
COPY utils.py  .

RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

CMD [ "app.handler" ]