from flask import Flask, request
from flask_restful import Api, Resource, reqparse
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
api = Api(app)
app.config.from_object("api.config.Config")
db = SQLAlchemy(app)

class Comment(db.Model):

    __tablename__ = 'submitted_comments'
    id = db.Column(db.Integer, primary_key=True, unique=True)
    comment = db.Column(db.String(256), unique=True, nullable=False)
    class_toxic = db.Column(db.Integer, unique=False, nullable=True)

    def __init__(self, comment, class_toxic):

        self.comment = comment
        self.class_toxic = class_toxic

class Comment_api(Resource):

    def get(self):

        # args_parser = reqparse.RequestParser()
        # args_parser.add_argument('email', type= str)

        # args = args_parser.parse_args()
        comment_= request.args['comment']
        try:
            comment_info = db.session.query(Comment).filter_by(comment=comment_).first()
            return {'Comment': comment_info.name, "Class": comment_info.email}
        
        except:
            return {'ERROR': "Coulden't find the Comment"}

        

    def post(self):

        # args_parser = reqparse.RequestParser()
        # args_parser.add_argument('email', type= str)
        # args_parser.add_argument('name', type= str)

        # args = args_parser.parse_args()
        
        comment_= request.form['comment']
        class_toxic_ = request.form['class_toxic']
        try:
            db.session.add(Comment(comment=comment_, class_toxic=class_toxic_))
            db.session.commit()
            return {'Comment': comment_, 'Class': class_toxic_}
        except Exception as exp:
            print(exp)
            return {'ERROR': "Couldn't insert email"}

api.add_resource(Comment_api, '/comment')