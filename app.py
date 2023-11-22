# app.py

from Prediction_model import RegularizedLinearRegression
import numpy as np

from flask import Flask, request, render_template

app = Flask(__name__)

def load_model():
    # Assuming 'model' is an instance of your trained model
    loaded_model = RegularizedLinearRegression()
    loaded_model.load_model("trained_model.npz")


    return loaded_model

# Gradient background color
background_color = "linear-gradient(to right, #ff8a00, #da1b60)"

# Your encoding function using Label Encoder
def encode_input(month, date, day, year, day_encoding, month_encoding):
    
    encoded_month = month_encoding[month] 
    encoded_day = day_encoding[day]
    
    encoded_input = np.array([encoded_month, 
                              int(date), 
                              encoded_day,
                              int(year)])
    
    #print("Encoded_shape: ", encoded_input.shape)

    return encoded_input.reshape(1, -1)


@app.route('/')
def home():
    return render_template('index.html', background_color=background_color)

# +
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Load the model and encoders
        loaded_model = load_model()

        # Get input values from the form
        month = request.form['month']
        date = request.form['date']
        day = request.form['day']
        year = request.form['year']
        
        day_encoding = {'Friday': 0, 'Monday': 1, 'Saturday': 2, 'Sunday': 3, 'Thursday': 4, 'Tuesday': 5, 'Wednesday': 6}
        month_encoding = {'April': 0, 'August': 1, 'December': 2, 'February': 3, 'January': 4, 'July': 5, 'June': 6, 'March': 7, 'May': 8, 'November': 9, 'October': 10, 'September': 11}

        
        # Validate inputs
        is_valid = True

        if day not in day_encoding:
            is_valid = False
    
        if month not in month_encoding:
            is_valid = False
    
        if date.isdigit() and 1 <= int(date) <= 31:
            date = int(date)
        
        else:
            is_valid = False
            
        
        if not is_valid:
            # Show error message
            error_msg = "Invalid input, please try again"
            return render_template('index.html', error=error_msg, background_color=background_color)

        # Encode and predict if valid
        encoded_input = encode_input(month, date, day, year, day_encoding, month_encoding)
        prediction = loaded_model.predict(encoded_input)

        return render_template('result.html', prediction=prediction[0], background_color=background_color)

        
        
# -

       # Encode the input
       #encoded_input = encode_input(month, date, day, year)

       # Make prediction
       #prediction = loaded_model.predict(encoded_input)

       #return render_template('result.html', prediction=prediction[0], background_color=background_color)


if __name__ == '__main__':
    app.run(debug=True)
