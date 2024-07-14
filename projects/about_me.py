import streamlit as st
import time
from forms.contact import contact_form
from streamlit_extras.colored_header import colored_header


st.set_page_config(layout="wide", page_title="Supradeep Danturti", page_icon="😊")


# ---Contact Form---
@st.experimental_dialog("Contact Me")
def contact():
    contact_form()


# ---Info---
c1, c2 = st.columns([0.3, 0.7], gap="small", vertical_alignment="center")

with c1:
    st.image("./assets/profile_picture.png", width=300)

with c2:
    colored_header(
        label="Hi! My name is Supradeep Danturti 👋🏼",
        description="Data Scientist | AI Engineer",
        color_name="violet-70",
    )
    st.markdown("<br>", unsafe_allow_html=True)

    social_icons_data = {
        "LinkedIn": ["https://www.linkedin.com/in/supradeepdanturti",
                     "https://cdn-icons-png.flaticon.com/512/174/174857.png"],
        "GitHub": ["https://github.com/supradeepdanturti",
                   "https://icon-library.com/images/github-icon-white/github-icon-white-6.jpg"]
    }

    social_icons_html = [
        (f"<a href='{social_icons_data[platform][0]}' target='_blank'><img class='social-icon' "
         f"src='{social_icons_data[platform][1]}' alt='{platform}' "
         f"style='width: 30px; height: 30px; margin-right: 10px;'></a>")
        for platform in social_icons_data
    ]

    st.markdown(" ".join(social_icons_html), unsafe_allow_html=True)

    if st.button("📬 Contact Me"):
        contact()

# --- SKILLS ---
st.write("\n")
colored_header(label="Skills", description="", color_name="violet-70")

skill_col_size = 4

info = {
    'skills': [
        "Python", "SQL", "Jupyter", "Numpy", "Pandas", "OpenCV", "Scikit-learn",
        "Pytorch", "Tensorflow", "Machine Learning", "Deep Learning",
        "ElasticSearch", "Kubernetes", "Docker", "Grafana", "Kibana", "Plotly",
        "Streamlit", "SpeechBrain"
    ]
}


def skill_tab():
    rows = len(info['skills']) // skill_col_size
    if len(info['skills']) % skill_col_size != 0:
        rows += 1

    skills = iter(info['skills'])

    for x in range(rows):
        columns = st.columns(skill_col_size)
        for index_ in range(skill_col_size):
            try:
                skill = next(skills)
                columns[index_].button(skill, key=skill)
            except StopIteration:
                break


with st.spinner(text="Loading section..."):
    skill_tab()

# --- Education & Experience ---
education_info = {
    "Master of Applied Computer Science, Concordia University": "2024",
    "Bachelor of Technology Computer Science, Centurion University of Technology and Management": "2022"
}

colored_header(label="Education", description="", color_name="violet-70")
st.markdown(
    """
    <style>
    .education-content {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
    }
    .education-date {
        font-style: italic;
        color: #777;
    }
    </style>
    """,
    unsafe_allow_html=True
)
for education, date in education_info.items():
    st.markdown(f"<div class='education-content'>{education}<div class='education-date'>{date}</div></div>",
                unsafe_allow_html=True)

colored_header(label="Experience", description="", color_name="violet-70")
st.markdown(
    """
    <style>
    .expander-content {
        display: flex;
        justify-content: space-between;
        width: 100%;
        align-items: flex-start;
    }
    .expander-date {
        flex-shrink: 0;
        margin-left: auto;
        font-style: italic;
        color: #777;
    }
    </style>
    """,
    unsafe_allow_html=True
)
gc = """
    Release Management Team
    - Engineered a comprehensive data visualization dashboard using Grafana, Kibana, Elasticsearch, and Python to monitor millions of daily transactions, resulting in a 20% decrease in deployment errors.
    - Developed and implemented a forecasting algorithm with 92% accuracy, utilizing Deep Learning techniques to accurately predict future trends for strategic decision-making in managing transactions.
    - Automated document creation processes with SQL and Python, improving productivity by 15% and reducing manual labor by 30% through streamlined daily monitoring procedures and ETL processes.
    """

mts_1 = """
    R&D Team
    - Developed a real-time attendance monitoring model for employees using machine learning/deep learning algorithms, leveraging Python, Node.js and OpenCV for data processing and analysis.
    - Contributed to the development of an ALPR system. Leveraged advanced Computer Vision techniques and to automatically detect and recognize license plates, streamlining processes and enhancing security measures for regulatory compliance.
    """

mts_2 = """
    MID Labs Innovation Team
    - Created a Proof of Concept (POC) for a video analytics prototype to automate the hiring process using Deep Learning, Computer Vision and Natural Language Processing.
    - Developed a cutting-edge prototype capable of analyzing candidate videos to extract crucial information, including facial expressions, body language, and speech patterns, utilizing Python and JSON for data processing.
    - Utilized this data-driven approach to assess candidates' qualifications and suitability for job.
    """

with st.expander("Data Analyst Intern, GeoComply", expanded=False):
    st.markdown("<div class='expander-content'><div></div><div class='expander-date'>May 2023 - Aug 2023</div></div>",
                unsafe_allow_html=True)
    st.write(gc)

with st.expander("Machine Learning Engineer, MotherSon Technology Services Jan 2022 - Aug 2022", expanded=False):
    st.markdown("<div class='expander-content'><div></div><div class='expander-date'>Jan 2022 - Aug 2022</div></div>",
                unsafe_allow_html=False)
    st.write(mts_1)

with st.expander("Machine Learning Intern, MotherSon Technology Services May 2021 - Dec 2021", expanded=False):
    st.markdown("<div class='expander-content'><div></div><div class='expander-date'>May 2021 - Dec 2021</div></div>",
                unsafe_allow_html=False)
    st.write(mts_2)

colored_header(label="Projects", description="", color_name="violet-70")
prj1, prj2, prj3 = st.columns([0.35, 0.30, 0.35], gap="small", vertical_alignment="bottom")

image_urls = [
    "assets/ConvAIProject.jpeg",
    "assets/AgeClassification.png",
    "https://example.com/image3.jpg"
]

titles = [
    "Speaker Detection & Counter",
    "Human Age Classification",
    "Project 3"
]

with prj1:
    st.image(image_urls[0], width=300)
    st.page_link("projects/SpeakerCounter_dir/SpeakerCounter_main.py", label="Speaker Counter🔊🔢", use_container_width=True)
with prj2:
    st.image(image_urls[1], width=300)
    st.page_link("https://github.com/SupradeepDanturti/COMP6721_Winter2023_GroupG", label="Age Classification Using CNNs", use_container_width=True)
with prj3:
    st.page_link("https://github.com/SupradeepDanturti/",
                 label="Adding more projects... meanwhile checkout my github repo", use_container_width=True)


hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)
