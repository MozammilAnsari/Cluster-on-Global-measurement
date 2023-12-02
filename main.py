import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import streamlit as st
import pickle
from sklearn.manifold import TSNE


list_of_country = ['Algeria',
 'Angola',
 'Benin',
 'Botswana',
 'Burkina Faso',
 'Burundi',
 'Cameroon',
 'Central African Republic',
 'Chad',
 'Comoros',
 'Congo, Dem. Rep.',
 'Congo, Rep.',
 "Cote d'Ivoire",
 'Djibouti',
 'Egypt, Arab Rep.',
 'Equatorial Guinea',
 'Eritrea',
 'Ethiopia',
 'Gabon',
 'Gambia, The',
 'Ghana',
 'Guinea',
 'Guinea-Bissau',
 'Kenya',
 'Lesotho',
 'Liberia',
 'Libya',
 'Madagascar',
 'Malawi',
 'Mali',
 'Mauritania',
 'Mauritius',
 'Morocco',
 'Mozambique',
 'Namibia',
 'Niger',
 'Nigeria',
 'Rwanda',
 'Sao Tome and Principe',
 'Senegal',
 'Seychelles',
 'Sierra Leone',
 'Somalia',
 'South Africa',
 'South Sudan',
 'Sudan',
 'Swaziland',
 'Tanzania',
 'Togo',
 'Tunisia',
 'Uganda',
 'Zambia',
 'Zimbabwe',
 'Afghanistan',
 'Armenia',
 'Azerbaijan',
 'Bangladesh',
 'Bhutan',
 'Brunei Darussalam',
 'Cambodia',
 'China',
 'Georgia',
 'Hong Kong SAR, China',
 'India',
 'Indonesia',
 'Japan',
 'Kazakhstan',
 'Korea, Dem. Rep.',
 'Korea, Rep.',
 'Kyrgyz Republic',
 'Lao PDR',
 'Macao SAR, China',
 'Malaysia',
 'Maldives',
 'Mongolia',
 'Myanmar',
 'Nepal',
 'Pakistan',
 'Philippines',
 'Singapore',
 'Sri Lanka',
 'Tajikistan',
 'Thailand',
 'Timor-Leste',
 'Turkmenistan',
 'Uzbekistan',
 'Vietnam',
 'Albania',
 'Andorra',
 'Austria',
 'Belarus',
 'Belgium',
 'Bosnia and Herzegovina',
 'Bulgaria',
 'Croatia',
 'Cyprus',
 'Czech Republic',
 'Denmark',
 'Estonia',
 'Faeroe Islands',
 'Finland',
 'France',
 'Germany',
 'Greece',
 'Hungary',
 'Iceland',
 'Ireland',
 'Isle of Man',
 'Italy',
 'Kosovo',
 'Latvia',
 'Liechtenstein',
 'Lithuania',
 'Luxembourg',
 'Macedonia, FYR',
 'Malta',
 'Moldova',
 'Monaco',
 'Montenegro',
 'Netherlands',
 'Norway',
 'Poland',
 'Portugal',
 'Romania',
 'Russian Federation',
 'San Marino',
 'Serbia',
 'Slovak Republic',
 'Slovenia',
 'Spain',
 'Sweden',
 'Switzerland',
 'Turkey',
 'Ukraine',
 'United Kingdom',
 'Bahrain',
 'Iran, Islamic Rep.',
 'Iraq',
 'Israel',
 'Jordan',
 'Kuwait',
 'Lebanon',
 'Oman',
 'Qatar',
 'Saudi Arabia',
 'Syrian Arab Republic',
 'United Arab Emirates',
 'Yemen, Rep.',
 'American Samoa',
 'Australia',
 'Fiji',
 'French Polynesia',
 'Guam',
 'Kiribati',
 'Marshall Islands',
 'Micronesia, Fed. Sts.',
 'New Caledonia',
 'New Zealand',
 'Papua New Guinea',
 'Samoa',
 'Solomon Islands',
 'Tonga',
 'Vanuatu',
 'Antigua and Barbuda',
 'Argentina',
 'Aruba',
 'Bahamas, The',
 'Barbados',
 'Belize',
 'Bermuda',
 'Bolivia',
 'Brazil',
 'Canada',
 'Cayman Islands',
 'Chile',
 'Colombia',
 'Costa Rica',
 'Cuba',
 'Curacao',
 'Dominica',
 'Dominican Republic',
 'Ecuador',
 'El Salvador',
 'Greenland',
 'Grenada',
 'Guatemala',
 'Guyana',
 'Haiti',
 'Honduras',
 'Jamaica',
 'Mexico',
 'Nicaragua',
 'Panama',
 'Paraguay',
 'Peru',
 'Puerto Rico',
 'Sint Maarten (Dutch part)',
 'St. Kitts and Nevis',
 'St. Lucia',
 'St. Martin (French part)',
 'St. Vincent and the Grenadines',
 'Suriname',
 'Trinidad and Tobago',
 'Turks and Caicos Islands',
 'United States',
 'Uruguay',
 'Venezuela, RB',
 'Virgin Islands (U.S.)']
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
    st.title('Clusters on Global Development Measurement')
    st.sidebar.header("Input Data")

    # Input for the new point
    birth_rate = st.sidebar.number_input("Birth Rate", min_value=0.0, step=0.1, max_value=1.0)
    business_tax_rate = st.sidebar.number_input("Business Tax Rate", min_value=0.0, step=0.1, max_value=1.00)
    co2_emissions = st.sidebar.number_input("CO2 Emissions", min_value=0.0, step=1.0)
    days_to_start_business = st.sidebar.number_input("Days to Start Business", min_value=0, step=1)
    health_exp_gdp = st.sidebar.number_input("Health Expenditure GDP", min_value=0.0, step=0.1)
    hours_to_do_tax = st.sidebar.number_input("Hours to do Tax", min_value=0, step=1)
    lending_interest = st.sidebar.number_input("Lending Interest", min_value=0.0, step=0.1)
    
    #available_countries = ["Country1", "Country2", "Country3"]  # Add your list of countries
    country = st.sidebar.selectbox("Select Country", list_of_country)

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



