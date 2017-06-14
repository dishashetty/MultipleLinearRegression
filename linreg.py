#Disha Karunakar Shetty
# id: 800966204
# dshetty1@uncc.edu

# linreg.py
#
# Standalone Python/Spark program to perform linear regression.
# Performs linear regression by computing the summation form of the
# closed form expression for the ordinary least squares estimate of beta.
# 
# TODO: Write this.
# 
# Takes the yx file as input, where on each line y is the first element 
# and the remaining elements constitute the x.
#
# Usage: spark-submit linreg.py <inputdatafile>
# Example usage: spark-submit linreg.py yxlin.csv
#
#

import sys
import numpy as np

from pyspark import SparkContext
from numpy.linalg import inv

#function to calculate x*x_transpose
def X_Xt(tempLine):
 tempLine[0]=1.0
 x1=np.array(tempLine).astype('float')
 x2=np.asmatrix(x1).T
 xxt=np.dot(x2,x2.T)
 return xxt

#function to calculate x*y
def X_Y(tempLine):
 y=float(tempLine[0])
 tempLine[0]=1.0
 xy1=np.array(tempLine).astype('float')
 xy2=np.asmatrix(xy1).T
 xy=np.multiply(xy2,y)
 return xy

if __name__ == "__main__":
  if len(sys.argv) !=2:
    print >> sys.stderr, "Usage: linreg <datafile>"
    exit(-1)

  sc = SparkContext(appName="LinearRegression")

  # Input yx file has y_i as the first element of each line 
  # and the remaining elements constitute x_i
  yxinputFile = sc.textFile(sys.argv[1])

  yxlines = yxinputFile.map(lambda line: line.split(','))
  yxfirstline = yxlines.first()
  yxlength = len(yxfirstline)
 
  
  # dummy floating point array for beta to illustrate desired output format
  beta = np.zeros(yxlength, dtype=float)

  #passing each line to the function X_Xt which returns x*x_transpose and creates key value pairs
  xxtValue =yxlines.map(lambda tempLine:("X_Xt", X_Xt(tempLine)))

  #passing key values pairs to the reducer
  xxtReducer=xxtValue.reduceByKey(lambda a,b: np.add(a,b))
 
  #getting value of summation of x*x_transpose
  part1=np.asmatrix(xxtReducer.map(lambda tempLine: tempLine[1]).collect()[0])

  #passing each line to the function X_Y which returns x*y and creates key value pairs
  xyValue =yxlines.map(lambda tempLine:("X_Y", X_Y(tempLine)))

  #passing key values pairs to the reducer
  xyReducer=xyValue.reduceByKey(lambda a,b: np.add(a,b))

  #getting value of the summation of x*y 
  part2=np.asmatrix(xyReducer.map(lambda tempLine: tempLine[1]).collect()[0])
  

  #dot product of (summation(x*xt))inverse and summation(y*x) to get beta
  result=np.dot(inv(part1),part2)
  beta=np.array(result).tolist()
  
  # print the linear regression coefficients in desired output format
  print "beta: "
  for coeff in beta:
      print coeff[0]

  sc.stop()
