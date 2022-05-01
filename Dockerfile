FROM amazon/aws-lambda-python:3.8

# Copy function code
COPY app.py ${LAMBDA_TASK_ROOT}
ADD checkpoint ./checkpoint
ADD models ./models

# Install the function's dependencies using file requirements.txt
# from your project folder.
RUN yum install -y mesa-libGL

COPY requirements.txt  .
COPY build_custom_model.py  .
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.handler" ]