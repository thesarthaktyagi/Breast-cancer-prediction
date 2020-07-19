from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle


if __name__ == "__main__":
    df = pd.read_csv('data.csv')
    df = df.dropna(axis=1)

    # Creating an object of LabelEncoder class
    labelencoder_Y = LabelEncoder()

    # Encoding the diagnosis column of the df
    df.iloc[:, 1] = labelencoder_Y.fit_transform(df.iloc[:, 1].values)

    X = df.iloc[:, 2:31].values
    Y = df.iloc[:, 1].values

    # Splitting the data into training and testing set

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.25, random_state=0)

    # Feature Scaling

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    forest = RandomForestClassifier(
        n_estimators=10, criterion='entropy', random_state=0)
    forest.fit(X_train, Y_train)

    # creating the artifact
    file = open('model.pkl', 'wb')

    # dump information
    pickle.dump(forest, file)
    pickle.dump(X_train, file)
    file.close()
