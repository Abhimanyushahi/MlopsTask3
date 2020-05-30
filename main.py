#!/usr/bin/env python
# coding: utf-8

# In[3]:


import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.preprocessing import image 


# In[4]:


(x_train, y_train) , (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(100000,784)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


# In[5]:


accuracy = 0
count = 1
learning_rate = 0.1
epoch = 12


# In[ ]:


while accuracy < .90 :
    model = Sequential()
    for i in range(count) :
        model.add(Dense(units=128,activation="relu",input_shape=(784,)))
    
    count = count +1
    print("count is ", counter)
    model.add(Dense(units=10, activation="softmax"))
    learning_rate = learnig_rate/10
    print("learning rate is :", learning_rate)
    model.compile(optimizer=SGD(learning_rate),loss="categorical_crossentropy",metrics=["accuracy"])
    model.fit(x_train,y_train,batch_size=32,epochs=epoch,verbose=1)
    model.summary()
    Accuracy = model.evaluate(x=x_test,y=y_test,batch_size=32)
    print("Accuracy : ",Accuracy[1])
    accuracy = Accuracy[1]
    print(accuracy)
    

