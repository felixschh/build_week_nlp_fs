from flask import Flask, render_template, request
from neuralnetwork.model import Classifier
import torch
import torch.nn.functional as F
from utils.preproccesing import clean_text, fit_comment


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

    return render_template('index.html', prediction = pred)


if __name__ == '__main__':
    app.run("0.0.0.0", port=3000, debug=True)