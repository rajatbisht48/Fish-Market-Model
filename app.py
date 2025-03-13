from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model using joblib
model = joblib.load(r'C:\Users\rajat\Desktop\Lab4\fish_model.pkl')
print("Model loaded successfully!")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the input features from the form
        species = int(request.form["species"])
        length1 = float(request.form["length1"])
        length2 = float(request.form["length2"])
        length3 = float(request.form["length3"])
        height = float(request.form["height"])
        width = float(request.form["width"])

        # Prepare the data for prediction (ensure it's in the correct shape)
        data = np.array([[species, length1, length2, length3, height, width]])

        # Make prediction using the model
        prediction = model.predict(data)

        # Render the result on the web page
        return render_template("index.html", prediction=f"Predicted Weight: {prediction[0]:.2f} grams")

    except Exception as e:
        return render_template("index.html", error=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
