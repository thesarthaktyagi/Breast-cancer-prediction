from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
from flask import Flask, render_template, request
app = Flask(__name__)

file = open('model.pkl', 'rb')
forest = pickle.load(file)
file.close()
@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        myDict = request.form

        Mean_Radius = myDict['Mean_Radius']
        Mean_Texture = myDict['Mean_Texture']
        Mean_Perimeter = myDict['Mean_Perimeter']
        Mean_Area = myDict['Mean_Area']
        Mean_Smoothness = myDict['Mean_Smoothness']
        Mean_Compactness = myDict['Mean_Compactness']
        Mean_Concavity = myDict['Mean_Concavity']
        Mean_Concave_Points = myDict['Mean_Concave_Points']
        Mean_Symmetry = myDict['Mean_Symmetry']
        Mean_Fractal_Dimension = myDict['Mean_Fractal_Dimension']
        SE_Radius = myDict['SE_Radius']
        SE_Texture = myDict['SE_Texture']
        SE_Perimeter = myDict['SE_Perimeter']
        SE_Area = myDict['SE_Area']
        SE_Smoothness = myDict['SE_Smoothness']
        SE_Compactness = myDict['SE_Compactness']
        SE_Concavity = myDict['SE_Concavity']
        SE_Concave_Points = myDict['SE_Concave_Points']
        SE_Symmetry = myDict['SE_Symmetry']
        SE_Fractal_Dimension = myDict['SE_Fractal_Dimension']
        Worst_Radius = myDict['Worst_Radius']
        Worst_Texture = myDict['Worst_Texture']
        Worst_Perimeter = myDict['Worst_Perimeter']
        Worst_Area = myDict['Worst_Area']
        Worst_Smoothness = myDict['Worst_Smoothness']
        Worst_Compactness = myDict['Worst_Compactness']
        Worst_Concavity = myDict['Worst_Concavity']
        Worst_Concave_Points = myDict['Worst_Concave_Points']
        Worst_Symmetry = myDict['Worst_Symmetry']

        input_features = [Mean_Radius,  Mean_Texture, Mean_Perimeter, Mean_Area, Mean_Smoothness, Mean_Compactness, Mean_Concavity, Mean_Concave_Points,
                          Mean_Symmetry, Mean_Fractal_Dimension, SE_Radius, SE_Texture, SE_Perimeter, SE_Area, SE_Smoothness, SE_Compactness, SE_Concavity, SE_Concave_Points,
                          SE_Symmetry, SE_Fractal_Dimension, Worst_Radius, Worst_Texture, Worst_Perimeter, Worst_Area, Worst_Smoothness, Worst_Compactness, Worst_Concavity,
                          Worst_Concave_Points, Worst_Symmetry]

        pred = forest.predict([input_features])
        print(pred[0])
        if pred[0] == 0:
            news = "Good News! Pateint's health is good."

        else:
            news = 'Bad News! Patient have Malignant Cancer'

        return render_template('result.html', result=news)
    return render_template('index.html')
    # return 'Hello, World!' + str(pred[0])


if __name__ == "__main__":
    app.run(debug=True)
