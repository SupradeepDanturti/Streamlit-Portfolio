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

Chatbot = st.Page(
    title="Chatbot",
    page="projects/Chatbot/Chatbot.py",
)
SpeakerCounter = st.Page(
    title="SpeakerCounter",
    page="projects/SpeakerCounter_dir/SpeakerCounter_main.py",
)

# Page Navs

navs = st.navigation(
    {
        "Info": [about_page],
        "Projects": [Chatbot, SpeakerCounter, Custom_Data_Visualizer, sftrees],
    }
)
navs.run()
