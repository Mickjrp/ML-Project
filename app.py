import pandas as pd
import joblib
from sklearn.calibration import LabelEncoder

# Load the saved model
model = joblib.load('FinalModel.joblib')

# Load Data
excel_file_path = './UserData/user_input.xlsx'

sheet_names_nd = ['nd' + str(i) for i in range(1, 11)]
sheet_names_d = ['d' + str(i) for i in range(1, 11)]
all_sheet_names = sheet_names_nd + sheet_names_d

user_data_dict = {}
for sheet_name in all_sheet_names:
    user_data_dict[sheet_name] = pd.read_excel(excel_file_path, sheet_name=sheet_name)

#print(user_data_dict.keys())

#----------------   Prepare Data  --------------------

def encode(data):
    non_numeric_columns = data.select_dtypes(exclude=['number']).columns

    if not non_numeric_columns.empty:
        label_encoder = LabelEncoder()

        for column in non_numeric_columns:
            data[column] = label_encoder.fit_transform(data[column])
        return data
    else:
        return ValueError("No non-numeric columns found.")



#----------------   predictive system --------------------
# Function to predict
def predict(user_data):
    
    # encode
    encode_data = encode(user_data)

    # Make prediction
    prediction = model.predict(encode_data)
    return "Depressed" if prediction[0] == 1 else "Not Depressed"


for user, data in user_data_dict.items():
    result = predict(data)
    print(f"Result for user {user}: {result}")


# Predict
# result = predict(user_data_dict['nd1'])
# print("Result:", result)

