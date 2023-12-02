import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import streamlit as st
import pickle
from sklearn.manifold import TSNE

# dataframe
data = pickle.load(open('df.pkl', 'rb'))
print(data.head())

# Label encoder for country
encoder = pickle.load(open('Label_encoder.pkl','rb'))

#model 
model = pickle.load(open('kmeans.pkl', 'rb'))

# TSNE for transforming data
tsne = pickle.load(open('tsne.pkl', 'rb'))

# Define the Streamlit app
def main():
    # Set the app title
    st.title('Clusters on global development measurement')
    st.sidebar.header("Input Data")

    # Input for the new point
    birth_rate = st.sidebar.number_input("Birth Rate", min_value=0.0, step=0.1)
    business_tax_rate = st.sidebar.number_input("Business Tax Rate", min_value=0.0, step=0.1)
    co2_emissions = st.sidebar.number_input("CO2 Emissions", min_value=0.0, step=1.0)
    days_to_start_business = st.sidebar.number_input("Days to Start Business", min_value=0, step=1)
    health_exp_gdp = st.sidebar.number_input("Health Expenditure GDP", min_value=0.0, step=0.1)
    hours_to_do_tax = st.sidebar.number_input("Hours to do Tax", min_value=0, step=1)
    lending_interest = st.sidebar.number_input("Lending Interest", min_value=0.0, step=0.1)
    country = st.sidebar.text_input("Country")

    # Add a "Submit" button
    if st.sidebar.button("Submit"):
        try:
            country_encoded = encoder.transform([country])[0]
        except ValueError:
            # Handle the case where the label is unseen during training
            country_encoded = -1

        # Create a DataFrame with the input
        input_data = {
            'BirthRate': [birth_rate],
            'BusinessTaxRate': [business_tax_rate],
            'CO2Emissions': [co2_emissions],
            'DaystoStartBusiness': [days_to_start_business],
            'HealthExpGDP': [health_exp_gdp],
            'HourstodoTax': [hours_to_do_tax],
            'LendingInterest': [lending_interest],
            'Country_encoded': [country_encoded]
        }

        input_df = pd.DataFrame(input_data)

        # Display the input DataFrame
        st.subheader("Input DataFrame:")
        result_df = pd.concat([data, input_df], ignore_index=True)
        st.write(input_df)

        # Perform t-SNE transformation
        data_tsne = tsne.fit_transform(result_df)

        # Get the last point for prediction
        new_data = data_tsne[-1, :].reshape(1, -1)

        # Predict using the KMeans model
        result = model.predict(new_data)
        
        st.header("Predicted Cluster:" +"  "+ str(result[0]))

if __name__ == "__main__":
    main()



