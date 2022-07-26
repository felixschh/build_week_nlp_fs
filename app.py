from turtle import forward
from flask import Flask, render_template, request
# from neuralnetwork.model import Network
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

@app.route('/', methods=['GET'])
def basic_page():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    comment = request.form['comment']
    print(comment)
    # path = './uploads/' + comment.filename
    # comment.save(path)

    # model = Network()
    # model.load_state_dict(torch.load('./neuralnetwork/checkpoint_14.pth'))


    # Preprocessing the Comment to fit in Model


    # return f'The file was succesfully uploaded to: {path}'
    return comment

if __name__ == '__main__':
    app.run("0.0.0.0", port=3000, debug=True)