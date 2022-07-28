from turtle import forward
from flask import Flask, render_template, request
from neuralnetwork.model import Classifier
import torch
import torch.nn.functional as F
from utils.preproccesing import preprocessing, token_encoder, clean_text, padding


app = Flask(__name__)

@app.route('/', methods=['GET'])
def basic_page():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    sub_comment = request.form['comment']
    # print(comment)
    # path = './uploads/' + comment.filename
    # comment.save(path)

    # Preprocessing the Comment to fit in Model


    comment = ''

    model = Classifier(32, 300,16,16)
    model.load_state_dict(torch.load('./neuralnetwork/model_states/trained_state_95.pth'))
    labels = []
    with torch.no_grad():
        comment.resize_(comment.size()[0], 32 * 300) #300 is embedding-size
        output_test = model.forward(comment)
        # prediction_label = torch.argmax(output_test, dim=1)
        prediction_label = torch.sigmoid(output_test)
        # print(prediction_label)
        classes = prediction_label > 0.5
        result = torch.sum(classes == labels, dim= 1) == len(labels) 

    return f'The comment you submitted was classified as : {result}'

if __name__ == '__main__':
    app.run("0.0.0.0", port=3000, debug=True)