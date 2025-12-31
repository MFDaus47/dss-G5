# filename: app.py
import streamlit as st
import pandas as pd
import numpy as np

# Title
st.title("Dummy Streamlit Interface")

# Subtitle
st.subheader("This is a simple demo to try Streamlit features")

# Text input
name = st.text_input("Enter your name:")

# Number input
age = st.number_input("Enter your age:", min_value=0, max_value=100, value=25)

# Button
if st.button("Greet me"):
    st.write(f"Hello {name}! You are {age} years old.")

# Checkbox
show_data = st.checkbox("Show sample data")
if show_data:
    # Create a sample dataframe
    data = pd.DataFrame(
        np.random.randn(10, 3),
        columns=['Column A', 'Column B', 'Column C']
    )
    st.write(data)

# Line chart
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['X', 'Y', 'Z']
)
st.line_chart(chart_data)

# Slider example
slider_value = st.slider("Select a value", 0, 100, 50)
st.write(f"You selected: {slider_value}")


