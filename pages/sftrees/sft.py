import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import pydeck as pdk

st.title("SF Trees")
trees_df = pd.read_csv("pages/sftrees/trees.csv")
st.write(trees_df.head())
# ---Built in Graphing Functions---


st.write(
    "This app analyses trees in San Francisco using"
    " a dataset kindly provided by SF DPW"
)

df_dbh_grouped = pd.DataFrame(trees_df.groupby(["dbh"]).count()["tree_id"])
df_dbh_grouped.columns = ["tree_count"]
st.line_chart(df_dbh_grouped)
st.bar_chart(df_dbh_grouped)
st.area_chart(df_dbh_grouped)

# ---Built in Graphing pt 2---


st.write(trees_df.head())
np.bool = np.bool_
df_dbh_grouped = pd.DataFrame(trees_df.groupby(["dbh"]).count()["tree_id"])
df_dbh_grouped.columns = ["tree_count"]
df_dbh_grouped["new_col"] = np.random.randn(len(df_dbh_grouped)) * 500
st.line_chart(df_dbh_grouped)
st.bar_chart(df_dbh_grouped)
st.area_chart(df_dbh_grouped)

# ---Built in Map---


trees_df = trees_df.dropna(subset=["longitude", "latitude"])
trees_df = trees_df.sample(n=1000)
st.map(trees_df)

# Plotly


st.subheader("Plotly Chart")

fig = px.histogram(trees_df["dbh"])
st.plotly_chart(fig)

# Matplotlib and Seaborn


trees_df["age"] = (pd.to_datetime("today") - pd.to_datetime(trees_df["date"])).dt.days

st.subheader("Seaborn Chart")
fig_sb, ax_sb = plt.subplots()
ax_sb = sns.histplot(trees_df["age"])
plt.xlabel("Age (Days)")
st.pyplot(fig_sb)

st.subheader("Matploblib Chart")
fig_mpl, ax_mpl = plt.subplots()
ax_mpl = plt.hist(trees_df["age"])
plt.xlabel("Age (Days)")
st.pyplot(fig_mpl)

# Altair


st.title("SF Trees: Altair")

df_caretaker = trees_df.groupby(["caretaker"]).count()["tree_id"].reset_index()
df_caretaker.columns = ["caretaker", "tree_count"]
fig = alt.Chart(df_caretaker).mark_bar().encode(x="caretaker", y="tree_count")
st.altair_chart(fig)

# Altair pt 2

st.title("SF Trees: Altair")

fig = alt.Chart(trees_df).mark_bar().encode(x="caretaker", y="count(*):Q")
st.altair_chart(fig)

# PyDeck

st.title("SF Trees: PyDeck")

trees_df.dropna(how="any", inplace=True)

sf_initial_view = pdk.ViewState(latitude=37.77, longitude=-122.4, zoom=11, pitch=30)

hx_layer = pdk.Layer(
    "HexagonLayer",
    data=trees_df,
    get_position=["longitude", "latitude"],
    radius=100,
    extruded=True,
)

st.pydeck_chart(
    pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=sf_initial_view,
        layers=[hx_layer],
    )
)
