
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf

bc_data = pd.read_csv(r"C:\Users\syafi\Downloads\Breast Cancer\data.csv")

#%%


bc_data = bc_data.drop(['id','Unnamed: 32'],axis = 1)
bc_data['diagnosis'] = bc_data['diagnosis'].replace('M',1)
bc_data['diagnosis'] = bc_data['diagnosis'].replace('B',0)

#%%

bc_features = bc_data.copy().drop(['diagnosis'],axis=1)
bc_labels = bc_data[['diagnosis']]

#%%

from sklearn.model_selection import train_test_split

SEED = 12345
x_train,x_test,y_train,y_test = train_test_split(bc_features,bc_labels,test_size=0.2,random_state=SEED)

standardizer = sklearn.preprocessing.StandardScaler()
standardizer.fit(x_train)
x_train = standardizer.transform(x_train)
x_test = standardizer.transform(x_test)

# Data preparation is done

#%%

from tensorflow.keras import regularizers
nClass = len(np.unique(y_test))
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape = x_train.shape[1]),
    tf.keras.layers.Dense(32, activation="relu",kernel_regularizer=regularizers.l2(0.1)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation="relu",kernel_regularizer=regularizers.l2(0.1)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(16, activation="relu",kernel_regularizer=regularizers.l2(0.1)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(nClass, activation="softmax")
    ])

model.summary()

#%%
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics = ['accuracy'])
history = model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size = 32,epochs=50)


training_loss = history.history['loss']
val_loss = history.history['val_loss']
training_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = history.epoch


plt.plot(epochs,training_loss,label='Training loss')
plt.plot(epochs,val_loss,label='Validation loss')
plt.title('Training Loss vs Validation Loss')
plt.legend()
plt.figure()
plt.show()

plt.plot(epochs,training_acc,label='Training accuracy')
plt.plot(epochs,val_acc,label='Validation accuracy')
plt.title('Training Accuracy vs Validation Accuracy')
plt.legend()
plt.figure()
plt.show()
