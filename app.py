from turtle import forward
from flask import Flask, render_template, request
from neuralnetwork.model import Classifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.preproccesing import preprocessing, token_encoder, clean_text, padding, fit_comment


app = Flask(__name__)

@app.route('/', methods=['GET'])
def basic_page():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    sub_comment = request.form['comment']
    comment_cleaned = clean_text(sub_comment)

    comment = fit_comment(comment_cleaned)

    model = Classifier(32, 300,16,16)
    model.load_state_dict(torch.load('./neuralnetwork/model_states/trained_state_95.pth'))

    labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']



    with torch.no_grad():
        comment.resize_(comment.size()[0], 32 * 300) #300 is embedding-size

        pred = model.forward(comment).detach()
        pred = F.softmax(pred, dim=1).numpy()

        pred = pred.argmax(axis=1)

        pred = labels[int(pred[-1])]

        return f'The comment you submitted was classified as : {pred}'

        # output_test = model.forward(comment)
        # output_test = model(comment.unsqueeze(0))

        # prediction_label = torch.argmax(output_test, dim=1)
        # prediction_label = torch.sigmoid(output_test).view(comment.size(0), -1)
        # print(prediction_label)
        # classes = prediction_label > 0.5
        # result = torch.sum(classes, dim= 1) #== len(labels) 

    # return f'The comment you submitted was classified as : {len(classes)} | {labels}'
    # return f'output : {prediction_label > 0.5}'
    # return f'output-shape is: {prediction_label.size()}'

if __name__ == '__main__':
    app.run("0.0.0.0", port=3000, debug=True)