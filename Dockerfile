FROM python:3

ADD task_1_neat.py /
ADD get_data.py /
ADD rnns.py /
ADD task_1_model.py /
ADD wrappers.py /



RUN pip install tensorflow

CMD ["mkdir", "./logs"]
CMD ["mkdir", "./data"]

CMD ["python3", "./task_1_neat.py"]




