import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load teams and cities
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

# Load machine learning model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Streamlit app title
st.title('IPL WIN PREDICTOR')

# Create columns for layout
col1, col2, col3 = st.columns([1, 1, 2])

# User input widgets
with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))

with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

with col3:
    selected_city = st.selectbox('Select host city', sorted(cities))
    target = st.number_input('Target', min_value=1, step=1)

# Create columns for advanced options
col4, col5, col6 = st.columns(3)

# Advanced user input widgets with custom step values and maximum values
with col4:
    score = st.number_input('Score', min_value=0, step=1)

with col5:
    overs = st.number_input('Overs completed', min_value=0.0, max_value=20.0, step=0.1, format="%.1f")

with col6:
    wickets_left = st.number_input('Wickets out', min_value=0, max_value=11, step=1)

# Prediction button
if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets_left
    crr = score / overs
    rrr = (runs_left * 6) / balls_left

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_left],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    result = pipe.predict_proba(input_df)

    # Display predictions using a horizontal bar chart with custom font properties
    fig, ax = plt.subplots(figsize=(10, 6))
    teams_with_percentages = [f'{batting_team} - {round(result[0][1] * 100)}%', f'{bowling_team} - {round(result[0][0] * 100)}%']
    percentages = [result[0][1] * 100, result[0][0] * 100]
    ax.barh(teams_with_percentages, percentages, color=['green', 'red'])

    # Set custom font properties (Impact font)
    ax.set_xlim(0, 100)
    ax.set_title('Win Probability', fontsize=24, fontweight='bold', fontname='Impact')
    ax.tick_params(axis='both', which='major', labelsize=20)
    st.pyplot(fig)
