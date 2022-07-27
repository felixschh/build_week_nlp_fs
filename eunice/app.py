from flask import Flask, render_template, redirect, request
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///modeldata.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)



class Data(db.Model):
    # id = db.Column(db.Integer, primary_key=True)
    pclass = db.Column(db.String(150), nullable=False)
    sex = db.Column(db.String(150), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    sib = db.Column(db.Integer, nullable=False)
    parch = db.Column(db.Integer, nullable=False)
    embark= db.Column(db.String(150), nullable=False)
    



def dataa():
    pcl = request.form.get('Pclass')
    sex = request.form.get('Sex')
    age = request.form.get('Age')
    sib = request.form.get('SibSp')
    parch = request.form.get('Parch')
    embark = request.form.get('Embarked')
    Dataa = Data(pclass = pcl, sex = sex,age = age, sib = sib, parch = parch, embark = embark)
    db.session.add(Dataa)
    db.session.commit()
    return 'sucessful'

@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        dataa()
    else:
        return render_template('index.html')

    






if __name__ == '__main__':
    db.create_all()
    # create database and table including column names
    # pclass = db.Column(db.String(150), nullable=False)
    # sex = db.Column(db.String(150), nullable=False)
    # age = db.Column(db.Integer, nullable=False)
    # sib = db.Column(db.Integer, nullable=False)
    # parch = db.Column(db.Integer, nullable=False)
    # embark= db.Column(db.String(150), nullable=False)
    passenger = Data(pclass = 'first_class', sex = 'female', age = 20, sib = 3, parch = 2, embark = 'somewhere')
    db.session.add(passenger)
    db.session.commit()

    
    app.run(debug=True)