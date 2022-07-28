from flask import Flask, render_template, redirect, request
from flask_sqlalchemy import SQLAlchemy
import os.path



db = SQLAlchemy()

class Data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    comments = db.Column(db.String(150), nullable=False)
    prediction = db.Column(db.String(150), nullable=False)
    # severe_toxic = db.Column(db.Integer, nullable=False)
    # obscene = db.Column(db.Integer, nullable=False)
    # threat = db.Column(db.Integer, nullable=False)
    # insult= db.Column(db.String(150), nullable=False)

def dataa():
    comment = request.form.get('comment_text')
    # toxis  = 'toxic'
    # severe_toxic= 'severe_toxic'
    # obscene = 'obscene'
    # threat = 'threat'
    # insult = 'insult'
    pred = 'toxic'
    # Dataa = Data(comments = comment,toxis  = toxis, severe_toxic= severe_toxic, obscene = obscene, threat= threat, insult=insult)
    Dataa = Data(comments = comment, prediction=pred)
    db.session.add(Dataa)
    db.session.commit()
#     return pred

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///modeldata.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# This line was missing
db.init_app(app=app)

db.create_all(app=app)

@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        dataa()
    return render_template('index.html')




# if __name__ == '__main__':
# #     db.init_app(app=app)
# #     db.create_all(app=app)
#     app.run(debug=True)