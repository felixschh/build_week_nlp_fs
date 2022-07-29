FROM python:3

COPY ./requirements.txt /toxic_comments/requirements.txt

WORKDIR /mnist-classifier

RUN pip3 install -r requirements.txt

COPY . /toxic_comments

ENTRYPOINT [ "python3" ]

CMD [ "app.py" ]