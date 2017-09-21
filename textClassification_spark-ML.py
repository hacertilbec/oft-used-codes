from pyspark import SparkContext
sc = SparkContext()

# read file and turn it into an RDD
data = sc.textFile('...')
# change data regarding to the purpose
map_function = lambda x: ...
# return RDD into dataframe
df = data.map(map_function).toDF()

# Digitazing sentences (tf-idf used in this example) and labels
from pyspark.ml.feature import HashingTF, IDF, Tokenizer

sentenceColumn, labelColumn = "...", "..."
tokenizer = Tokenizer(inputCol=sentenceCol, outputCol="words")
wordsData = tokenizer.transform(df)

hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(wordsData)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol="labelColumn", outputCol="label")
indexed = indexer.fit(rescaledData).transform(rescaledData)

newData = indexed['features', 'label']
newData.show()

# split data
(trainingData, testData) = newData.randomSplit([0.7, 0.3])

# Classification

from pyspark.ml.classification import LogisticRegression, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# instantiate the base classifier.
lr = LogisticRegression(maxIter=10, tol=1E-6, fitIntercept=True)

# instantiate the One Vs Rest Classifier.
ovr = OneVsRest(classifier=lr)

# train the multiclass model.
ovrModel = ovr.fit(trainingData)

# score the model on test data.
predictions = ovrModel.transform(testData)

# obtain evaluator.
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")

# compute the classification error on test data.
accuracy = evaluator.evaluate(predictions)
print("Test Error : " + str(1 - accuracy))
