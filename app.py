import streamlit as st
# Page info
about_page = st.Page(
    title="About",
    page="pages/about_me.py",
    icon="😊",
    default=True,
)
Custom_Data_Visualizer = st.Page(
    title="Custom Data Visualizer",
    page="pages/customdatavisualizer/customdatavisualizer.py",
)

sftrees = st.Page(
    title="San Francisco Trees",
    page="pages/sftrees/sft.py",
)

HealthInsurance = st.Page(
    title="Health Insurance",
    page="pages/HealthInsuranceModel/HealthInsurance.py",
)
sample2 = st.Page(
    title="sample2",
    page="pages/sample2.py",
)

# Page Navs
#
# navs = st.navigation(pages=[about_page])

navs = st.navigation(
    {
        "Info": [about_page],
        "Projects": [Custom_Data_Visualizer, sftrees, HealthInsurance, sample2],
    }
)
navs.run()
