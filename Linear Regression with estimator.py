 import tensorflow as tensorflow
import numpy as np 

feature_columns = [tf.feature_column.numeric_column("x",shape =[1])]
estimator = tf.estimator.LinearRegressor(feature_columns = feature_columns)

x_train = np.array([1.3,3.4,5.4,3.0])
y_train = np.array([0.3,0.4,0.5,0.6])

x_test = np.array([-3.0,9.4,4.54,3.4])
y_test = np.array([-3,3.8,0.4,0.6])

input_function = tf.estimator.inputs.numpy_input_fn({"x":x_train},y_train,batch_size=4,num_epochs=None,shuffle-True)
train_input_function = tf.estimator.inputs.numpy_input_fn({"x":x_train},y_train,batch_size=4,num_epochs=1000,shuffle=False)
test_input_fn = tf.estimator.inputs.numpy_input_fn({"x":x_test},y_test,batch_size=4,num_epochs=1000,shuffle=False)

estimator.train(input_fn=input_function,steps=1000)

train_metrices = estimator.evaluate(input_fn=train_input_function)
test_metrices = estimator.evaluate(input_fn = test_input_fn)

print("train metrices: %r"% train_metrices)
print("test_metrices: %r"% test_metrices)
