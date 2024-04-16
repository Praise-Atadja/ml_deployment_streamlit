import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import joblib


# Load the model and encoder
model = joblib.load("random_forest.pkl")

# Scaling function
def min_max_scale_numerical_features(features, features_numerical):
    scaler = MinMaxScaler()
    features_minmax_transform = pd.DataFrame(data=features)
    features_minmax_transform[features_numerical] = scaler.fit_transform(
        features[features_numerical])
    return features_minmax_transform

# Encoding function
def encode_categorical_columns(features_minmax_transform):
    """
    Encodes categorical columns in the features_minmax_transform DataFrame using label encoding.
    """
    le = LabelEncoder()
    for column in features_minmax_transform.columns:
        if not pd.api.types.is_numeric_dtype(features_minmax_transform[column]):
            features_minmax_transform[column] = le.fit_transform(
                features_minmax_transform[column])
    return features_minmax_transform

# Preprocess features function
def preprocess_features(input_data):
    features_numerical = ['age']
    input_data_scaled = min_max_scale_numerical_features(
        input_data, features_numerical)
    input_data_encoded = encode_categorical_columns(input_data_scaled)
    return input_data_encoded

# Prediction function
def predict_input(input_data):
    # Preprocess the input data
    input_data_transformed = preprocess_features(input_data)

    # Make prediction
    prediction = model.predict(input_data_transformed)

    # Output the prediction
    return prediction


def main():
    st.title("Welcome to CognitiveQuest!")
    html_temp = """
    <div style="background:#025246 ;padding:10px"; margin-bottom: -2000px">
    <h2 style="color:white;text-align:center;">Your Awesome Autism Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

# Collect user input
    st.subheader("Please Fill in Your Details")
    A1_Score = st.selectbox("A1 Score", [0, 1])
    A2_Score = st.selectbox("A2 Score", [0, 1])
    A3_Score = st.selectbox("A3 Score", [0, 1])
    A4_Score = st.selectbox("A4 Score", [0, 1])
    A5_Score = st.selectbox("A5 Score", [0, 1])
    A6_Score = st.selectbox("A6 Score", [0, 1])
    A7_Score = st.selectbox("A7 Score", [0, 1])
    A8_Score = st.selectbox("A8 Score", [0, 1])
    A9_Score = st.selectbox("A9 Score", [0, 1])
    A10_Score = st.selectbox("A10 Score", [0, 1])

    age = st.selectbox("Age", list(range(4, 64)))
    gender = st.radio("Gender", ["Male", "Female"])
    ethnicity_options = [
        "White-European", "Latino", "Others", "Black", "Asian", "Middle Eastern",
        "Pasifika", "South Asian", "Hispanic", "Turkish"
    ]
    ethnicity = st.selectbox("Ethnicity", ethnicity_options)
    jaundice = st.radio("Jaundice history", ["Yes", "No"])
    autism = st.radio("Autism diagnosis", ["Yes", "No"])
    country_of_res_options = [
        "United States", "Brazil", "Spain", "Egypt", "New Zealand", "Bahamas",
        "Burundi", "Austria", "Argentina", "Jordan", "Ireland", "United Arab Emirates",
        "Afghanistan", "Lebanon", "United Kingdom", "South Africa", "Italy", "Pakistan",
        "China", "Australia", "Canada", "Saudi Arabia", "France", "Sierra Leone", "Ethiopia",
        "Mexico", "Netherlands", "Hong Kong", "Iceland", "Russia", "Belgium", "Costa Rica",
        "Germany", "Iran", "Viet Nam", "Sri Lanka", "Armenia", "Uruguay", "Italy",
        "Turkey", "Czech Republic", "Iraq", "Niger", "Indonesia", "Bolivia", "Angola",
        "Serbia", "Portugal", "Malaysia", "Sweden", "Philippines", "Egypt", "Malaysia", "AmericanSamoa",
        "Azerbaijan"
    ]
    country_of_res = st.selectbox("Country of residence", country_of_res_options)
    used_app_before = st.radio("Used app before", ["Yes", "No"])
    relation = st.selectbox("Relation", ["Health care professional", "Others", "Parent", "Relative", "Self"])
    age_range = st.selectbox("Age range", ["4-8", "9-13", "14-18", "19-23","24-28", "29-33", "34-38", "39-43", "44-48", "49-53", "54-58", "59-63"])


    input_data = pd.DataFrame({
        'A1_Score': [A1_Score],
        'A2_Score': [A2_Score],
        'A3_Score': [A3_Score],
        'A4_Score': [A4_Score],
        'A5_Score': [A5_Score],
        'A6_Score': [A6_Score],
        'A7_Score': [A7_Score],
        'A8_Score': [A8_Score],
        'A9_Score': [A9_Score],
        'A10_Score': [A10_Score],
        'age': [age],
        'gender': [gender],
        'ethnicity': [ethnicity],
        'jaundice': [jaundice],
        'autism': [autism],
        'country_of_res': [country_of_res],
        'used_app_before': [used_app_before],
        'relation': [relation],
        'age_range': [age_range]
    })

    if st.button("Submit Your Information"):
        # Make prediction
        prediction = predict_input(input_data)

        # Output the prediction
        if prediction == 1:
            st.success(
                "The prediction indicates presecence of Autism Spectrum Disorder (ASD).")

        else:
            st.success(
                "The prediction indicates no presecence of Autism Spectrum Disorder (ASD).")


if __name__ == "__main__":
    main()
