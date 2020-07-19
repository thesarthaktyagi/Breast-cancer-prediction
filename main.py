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

        Mean_Radius = int(myDict['Mean_Radius'])
        Mean_Texture = int(myDict['Mean_Texture'])
        Mean_Perimeter = int(myDict['Mean_Perimeter'])
        Mean_Area = int(myDict['Mean_Area'])
        Mean_Smoothness = int(myDict['Mean_Smoothness'])
        Mean_Compactness = int(myDict['Mean_Compactness'])
        Mean_Concavity = int(myDict['Mean_Concavity'])
        Mean_Concave_Points = int(myDict['Mean_Concave_Points'])
        Mean_Symmetry = int(myDict['Mean_Symmetry'])
        Mean_Fractal_Dimension = int(myDict['Mean_Fractal_Dimension'])
        SE_Radius = int(myDict['SE_Radius'])
        SE_Texture = int(myDict['SE_Texture'])
        SE_Perimeter = int(myDict['SE_Perimeter'])
        SE_Area = int(myDict['SE_Area'])
        SE_Smoothness = int(myDict['SE_Smoothness'])
        SE_Compactness = int(myDict['SE_Compactness'])
        SE_Concavity = int(myDict['SE_Concavity'])
        SE_Concave_Points = int(myDict['SE_Concave_Points'])
        SE_Symmetry = int(myDict['SE_Symmetry'])
        SE_Fractal_Dimension = int(myDict['SE_Fractal_Dimension'])
        Worst_Radius = int(myDict['Worst_Radius'])
        Worst_Texture = int(myDict['Worst_Texture'])
        Worst_Perimeter = int(myDict['Worst_Perimeter'])
        Worst_Area = int(myDict['Worst_Area'])
        Worst_Smoothness = int(myDict['Worst_Smoothness'])
        Worst_Compactness = int(myDict['Worst_Compactness'])
        Worst_Concavity = int(myDict['Worst_Concavity'])
        Worst_Concave_Points = int(myDict['Worst_Concave_Points'])
        Worst_Symmetry = int(myDict['Worst_Symmetry'])

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
