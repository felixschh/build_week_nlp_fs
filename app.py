from flask import Flask, render_template, request
from neuralnetwork.model import Classifier
import torch
import torch.nn.functional as F
from utils.preproccesing import clean_text, fit_comment
from flask_sqlalchemy import SQLAlchemy
import os

app = Flask(__name__)
db = SQLAlchemy()
# app.config.from_object("logic.web.api.config.Config")
# SQLALCHEMY_TRACK_MODIFICATIONS = False
# # SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite://')
# SQLALCHEMY_DATABASE_URI = 'sqlite://toxicity.sqlite'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///toxicity.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db = SQLAlchemy(app)
# db.init_app(app=app)
# db.create_all(app=app)

class Comment(db.Model):
    __tablename__ = 'submitted_comments'
    id = db.Column(db.Integer, primary_key=True, unique=True)
    comment = db.Column(db.String(256), unique=True, nullable=False)
    class_toxic = db.Column(db.Integer, unique=False, nullable=True)

db.init_app(app=app)
db.create_all(app=app)
    # def __init__(self, comment, class_toxic):

    #     self.comment = comment
    #     self.class_toxic = class_toxic

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
    
    submitted_comment = Comment(comment=sub_comment, class_toxic=pred)
    db.session.add(submitted_comment)
    db.session.commit()

    return render_template('index.html', prediction = pred)


if __name__ == '__main__':
    # app.run("0.0.0.0", port=3000, debug=True)
    app.run(port=3000, debug=True)