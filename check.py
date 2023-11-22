from Prediction_model import RegularizedLinearRegression
import numpy as np


def load_model():
    # Assuming 'model' is an instance of your trained model
    loaded_model = RegularizedLinearRegression()
    loaded_model.load_model("trained_model.npz")


    return loaded_model


# Your encoding function using Label Encoder
def encode_input(month, date, day, year):
    day_encoding = {'Friday': 0, 'Monday': 1, 'Saturday': 2, 'Sunday': 3, 'Thursday': 4, 'Tuesday': 5, 'Wednesday': 6}
    month_encoding = {'April': 0, 'August': 1, 'December': 2, 'February': 3, 'January': 4, 'July': 5, 'June': 6, 'March': 7, 'May': 8, 'November': 9, 'October': 10, 'September': 11}

    encoded_month = month_encoding[month] 
    encoded_day = day_encoding[day]
    
    encoded_input = np.array([encoded_month, 
                              int(date), 
                              encoded_day,
                              int(year)])
    
    #print("Encoded_shape: ", encoded_input.shape)

    return encoded_input.reshape(1, -1)



def predict():
    loaded_model = load_model()

    # Get input values from the form
    month = 'May'
    date = '12'
    day = 'Friday'
    year = '2022'

    # Encode the input
    encoded_input = encode_input(month, date, day, year)
    
    print("Encoded_shape: ", encoded_input.shape)    
                
    # Make prediction
    prediction = loaded_model.predict(encoded_input)
    
    print(prediction)

    #return render_template('result.html', prediction=prediction[0], background_color=background_color)

predict()