import streamlit as st

# Page info
about_page = st.Page(
    title="About",
    page="projects/about_me.py",
    icon="😊",
    default=True,
)
Custom_Data_Visualizer = st.Page(
    title="Custom Data Visualizer",
    page="projects/customdatavisualizer/customdatavisualizer.py",
)

sftrees = st.Page(
    title="San Francisco Trees",
    page="projects/sftrees/sft.py",
)

HealthInsurance = st.Page(
    title="Health Insurance",
    page="projects/HealthInsuranceModel/HealthInsurance.py",
)
Chatbot = st.Page(
    title="Chatbot",
    page="projects/Chatbot/Chatbot.py",
)
SpeakerCounter = st.Page(
    title="SpeakerCounter",
    page="projects/SpeakerCounter_dir/SpeakerCounter_main.py",
)
sample2 = st.Page(
    title="sample2",
    page="projects/sample2.py",
)

# Page Navs
#
# navs = st.navigation(projects=[about_page])

navs = st.navigation(
    {
        "Info": [about_page],
        "Projects": [SpeakerCounter, Chatbot, Custom_Data_Visualizer, sftrees, HealthInsurance, sample2],
    }
)
navs.run()
