###################################################################################
# Искуственный интеллект
# Лаба 2
# Ус Д. Д. Ктбо3-1
###################################################################################

import numpy as np
from tensorflow import keras
from keras import layers
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

#
#Данные для платформы Keggele
#

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


##################################################################
#
# Загрузка данных, разделение их на обучающую и текстовую выборку
#
##################################################################




train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")


X_train, X_validation, y_train, y_validation = train_test_split(train_df.iloc[:,1:], train_df['label'], test_size=0.2, random_state=21)
y_train = y_train.to_numpy()
y_validation = y_validation.to_numpy()

X_train = X_train.to_numpy()


import random


##################################################################
#
#Создание на основе данных логической регресси
#
##################################################################

logicalRegress = LogisticRegression()
logicalRegress.fit(X_train, y_train)
ped = logicalRegress.predict(X_validation)
print('Точность логической регрессии - {}'.format(logicalRegress.score(X_validation,y_validation)))

# print(X_train)



#####################################################################
#
#Трансформирование данных в удобный для обучения неронной сети формат
#
#####################################################################

# print(X_train)

X_train = X_train.reshape(X_train.shape[0], int(math.sqrt(X_train.shape[1])), -1)
# #
# print(X_train)
#
X_validation = X_validation.to_numpy()
# #
# # # print(X_validation)
# #
X_validation = X_validation.reshape(X_validation.shape[0], int(math.sqrt(X_validation.shape[1])), -1)
# #
# #
# # # print(X_validation)
# #
#


######################################################################
#
#Здесь происходит деление обучающей выборки на несколько массивов
#Так как размер нашего изображения 28x28, то нам удобно будет разделить его
#28 на x и 28 на y
#
######################################################################


X_test = test_df.loc[:, "pixel0":].to_numpy()
len_X_test = len(X_test)
X_test = X_test.reshape(len_X_test, 28, 28)
num_classes = 10
input_shape = (28, 28, 1)

#########################################################################
#
#Так как нам в данном прмере не важна градация цвета
#Важен лишь факт заженности пикселя, то для оптимизации сделаеи все заженные пиксели 1
#
##########################################################################
X_train = X_train.astype("float32") / 255
X_validation = X_validation.astype("float32") / 255
X_test = X_test.astype("float32") / 255
X_train = np.expand_dims(X_train, -1)
X_validation = np.expand_dims(X_validation, -1)
X_test = np.expand_dims(X_test, -1)




y_train = keras.utils.to_categorical(y_train, num_classes)
y_validation = keras.utils.to_categorical(y_validation, num_classes)



model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(X_validation, y_validation, verbose=0)
print("Test accuracy:", score[1])

y_pred = model.predict(X_test)
no_images=len(X_test)

# Display random Image
fig, ax = plt.subplots(figsize=(10, 10))

plt.imshow(X_test[0, :, :, 0], cmap='Greys', interpolation='nearest')
plt.show()

classes_x=np.argmax(y_pred, axis=1)
submit_this = pd.DataFrame({'Label': classes_x, 'ImageId': range(1, len(classes_x)+1)})
submit_this.to_csv("out.csv", index=False)