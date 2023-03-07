#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Basic').getOrCreate()
spark.conf.set("spark.sql.repl.eagerEval.enabled", True) # Property used to format output tables better
spark


# In[ ]:


from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import LinearRegressionModel
import pandas as pd


# In[ ]:


try: 
    path = input("File path: ")
    filetype = input("File type: ")
    df = spark.read.load(path, format=filetype, inferSchema=True, header=True)
except(ValueError, FileNotFoundError):
    print("You did not enter valid inputs.")


# In[ ]:


df.printSchema()


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


num_features = input("Number of features: ")


# In[ ]:


coef_var = []
for i in range(int(num_features)):
    coef_var.append(input("Feature name: "))
assembler = VectorAssembler(inputCols=coef_var, outputCol='features')


# In[ ]:


output = assembler.setHandleInvalid('skip').transform(df) 
#setHandleInvalid skip flag lets it skip null rows instead of having an error


# In[ ]:


output_col = input("Output column: ")


# In[ ]:


final_df = output.select('features', output_col)


# In[ ]:


train_data, test_data = final_df.randomSplit([0.7,0.3])


# In[ ]:


lm = LinearRegression(labelCol=output_col)


# In[ ]:


model = lm.fit(train_data)


# In[ ]:


pd.DataFrame({"Coefficients":model.coefficients}, index=coef_var)


# In[ ]:


res = model.evaluate(test_data)


# In[ ]:


res.residuals.show()


# In[ ]:


unlabeled_data = test_data.select('features')


# In[ ]:


predictions = model.transform(unlabeled_data)


# In[ ]:


predictions.show()


# In[ ]:


print("MAE: ", res.meanAbsoluteError)
print("MSE: ", res.meanSquaredError)
print("RMSE: ", res.rootMeanSquaredError)
print("R2: ", res.r2)
print("Adj R2: ", res.r2adj)


# In[ ]:


model.save("model")

