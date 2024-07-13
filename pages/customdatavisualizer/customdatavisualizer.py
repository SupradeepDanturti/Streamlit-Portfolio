import streamlit as st
import pandas as pd
import altair as alt
import base64

st.title("Custom Dataset Visualizer")


# Function to create download link for sample datasets
@st.cache_data()
def create_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href


# Sample datasets
sample_datasets = {
    "Penguins": pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv'),
    "Iris": pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'),
    "Tips": pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv'),
    "Titanic": pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv')
}

st.markdown("## Download Sample Datasets")
for name, df in sample_datasets.items():
    st.markdown(create_download_link(df, f"{name}.csv"), unsafe_allow_html=True)

# File uploader for CSV files
uploaded_file = st.file_uploader("Upload your CSV file (max 100 MB)", type="csv")

if uploaded_file is not None:
    # Load the uploaded CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Display the first few rows of the DataFrame
    st.write(df.head())

    st.markdown('Use this Streamlit app to visualize your data!')

    # Select plot type
    plot_type = st.selectbox("Select the plot type",
                             ["Scatter Plot", "Bar Plot", "Line Chart", "Area Chart", "Histogram", "Box Plot"])

    # Select columns for x and y variables
    x_var = st.selectbox("Select the x variable", df.columns)
    y_var = st.selectbox("Select the y variable", df.columns)

    # Optional: Select a column for color encoding (if it makes sense for the data)
    color_var = st.selectbox("Select the color variable (optional)", [None] + list(df.columns), index=0)

    # Create chart based on the selected plot type
    if plot_type == "Scatter Plot":
        chart = alt.Chart(df).mark_circle().encode(
            x=x_var,
            y=y_var,
            color=color_var if color_var else alt.value("blue")
        ).interactive()

    elif plot_type == "Bar Plot":
        chart = alt.Chart(df).mark_bar().encode(
            x=x_var,
            y=y_var,
            color=color_var if color_var else alt.value("blue")
        ).interactive()

    elif plot_type == "Line Chart":
        chart = alt.Chart(df).mark_line().encode(
            x=x_var,
            y=y_var,
            color=color_var if color_var else alt.value("blue")
        ).interactive()

    elif plot_type == "Area Chart":
        chart = alt.Chart(df).mark_area().encode(
            x=x_var,
            y=y_var,
            color=color_var if color_var else alt.value("blue")
        ).interactive()

    elif plot_type == "Histogram":
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(x_var, bin=True),
            y='count()',
            color=color_var if color_var else alt.value("blue")
        ).interactive()

    elif plot_type == "Box Plot":
        chart = alt.Chart(df).mark_boxplot().encode(
            x=x_var,
            y=y_var,
            color=color_var if color_var else alt.value("blue")
        ).interactive()

    # Display chart in Streamlit
    st.altair_chart(chart, use_container_width=True)
else:
    st.write("Please upload a CSV file to visualize.")
