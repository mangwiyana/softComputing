# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Inisialisasi Algo NN
# =================================
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# ==========================================
# END Inisialisasi Algo NN


X = dataset.iloc[:, 3:-1].values
Y = dataset.iloc[:, 13].values


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder_country_X = LabelEncoder()
labelencoder_gender_X = LabelEncoder()


X[:, 1] = labelencoder_country_X.fit_transform(X[:, 1])
X[:, 2] = labelencoder_gender_X.fit_transform(X[:, 2])

ohe = OneHotEncoder(categorical_features = [2])
X = ohe.fit_transform(X).toarray()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#scaling data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)