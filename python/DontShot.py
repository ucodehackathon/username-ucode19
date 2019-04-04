!wget --no-check-certificate \
   'URLZIP' -O \
   /tmp/football-data.zip
   
import os 
import numpy as np 
import tensorflow as tf
import pandas as pd
import zipfile

local_zip = '/tmp/football-data.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()
  
base_dir = '/tmp/football-data'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 
  
def LoadFromDir(base_dir):
     
    train_path = os.path.join(base_dir,'train/')
    train_batch = os.listdir(train_path)
    x_train = None
     
    # if data are in form of images
    for sample in train_batch:
      folderdir=os.listdir(train_path+sample)
      for subdir in folderdir:
        x = pd.read_csv(train_path+sample+'/'+subdir, engine='python').values
        
        # preprocessing if required
        x = np.delete(x, (0), axis=0)
        aux=np.zeros((len(x),11))
        target=int(sample[-2:])
        aux[:,target]=1
        x =np.hstack((x,aux))
        if(x_train is None):
          x_train=x
        else:
          x_train = np.concatenate((x_train,x))
     
    test_path = os.path.join(base_dir,'test')
    test_batch = os.listdir(test_path)
    x_test = None
     
    for sample in test_batch:
      folderdir=os.listdir(train_path+sample)
      for subdir in folderdir:
        x = pd.read_csv(train_path+sample+'/'+subdir, engine='python').values
        
        x=np.delete(x, (0), axis=0)
        aux=np.zeros((len(x),11))
        target=int(sample[-2:])
        aux[:,target]=1
        x =np.hstack((x,aux))
        # preprocessing if required
        if(x_test is None):
          x_test=x
        else:
          x_test = np.concatenate((x_test,x))
    return x_train, x_test
	
import matplotlib.pyplot as plt

plt.hist(x_train[:,0], bins=20)
plt.ylabel('Left-Accel-X [g]')
plt.show()

plt.hist(x_train[:,1], bins=20)
plt.ylabel('Left-Accel-Y [g]')
plt.show()

plt.hist(x_train[:,2], bins=20)
plt.ylabel('Left-Accel-Z [g]')
plt.show()

plt.hist(x_train[:,3], bins=20)
plt.ylabel('Left-Gyro-X [g/s]')
plt.show()

plt.hist(x_train[:,4], bins=20)
plt.ylabel('Left-Gyro-Y [g/s]')
plt.show()

plt.hist(x_train[:,5], bins=20)
plt.ylabel('Left-Gyro-Z [g/s]')
plt.show()

plt.hist(x_train[:,6], bins=20)
plt.ylabel('Right-Accel-X [g]')
plt.show()

plt.hist(x_train[:,7], bins=20)
plt.ylabel('Right-Accel-Y [g]')
plt.show()

plt.hist(x_train[:,8], bins=20)
plt.ylabel('Right-Accel-Z [g]')
plt.show()

plt.hist(x_train[:,9], bins=20)
plt.ylabel('Right-Gyro-X [g/s]')
plt.show()

plt.hist(x_train[:,10], bins=20)
plt.ylabel('Right-Gyro-Y [g/s]')
plt.show()

plt.hist(x_train[:,11], bins=20)
plt.ylabel('Right-Gyro-Z [g/s]')
plt.show()
plt.hist(x_train[:,12], bins=20)
plt.ylabel('Types')
plt.show()

from keras.models import Sequential  
from keras.layers import Dense  
from keras.layers import LSTM  
from keras.layers import Dropout  
from sklearn.preprocessing import MinMaxScaler

x_train,x_test = LoadFromDir(base_dir)
y_tr=x_train[:,-11:]
y_te=x_test[:,-11:]

x_train=tf.keras.utils.normalize(x_train,1,1)
x_test=tf.keras.utils.normalize(x_test,1,1)

x_tr = np.reshape(x_train[:,0:-12], x_train[:,0:-12].shape + (1,))
x_te = np.reshape(x_test[:,-11:], x_test[:,-11:].shape + (1,))

model = Sequential()
model.add(LSTM(50,input_shape=x_tr.shape[1:], return_sequences=True))  
model.add(Dropout(0.2))

#model.add(LSTM(units=50, return_sequences=True))  
#model.add(Dropout(0.2))

model.add(LSTM(units=50))  
model.add(Dropout(0.2)) 
model.add(Dense(11, activation='sigmoid'))
opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
model.compile(loss='categorical_crossentropy',
          optimizer=opt,
          metrics=['accuracy'])



history = model.fit(x_tr, y_tr,
                batch_size=100,
                epochs=100,
                verbose=1,
                validation_data=(x_te, y_te))
score = model.evaluate(x_te, y_te, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

	
	