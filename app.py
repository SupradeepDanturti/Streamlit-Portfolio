import streamlit as st

# Page info
about_page = st.Page(
    title="About",
    page="pages/about_me.py",
    icon="😊",
    default=True,
)
sample = st.Page(
    title="sample",
    page="pages/sample.py",
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
        "Projects": [sample, sample2],
    }
)
navs.run()
