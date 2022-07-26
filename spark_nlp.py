from email import header
import pyspark as ps
from pyspark.sql import SQLContext
from pyspark.ml.feature import CountVectorizer, Tokenizer,HashingTF, IDF
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


sc = ps.SparkContext.getOrCreate()
sql = SQLContext()

train = sql.read.format('com.databricks.spark.csv').options(header = False, inferSchema = True).load('jigsaw-toxic-comment-classification-challenge/train.csv/train.csv')
test = sql.read.format('com.databricks.spark.csv').options(header = False, inferSchema = True).load('jigsaw-toxic-comment-classification-challenge/test.csv/test.csv')

print(train.printSchema())