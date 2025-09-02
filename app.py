import streamlit as st
import pandas as pd
import requests
import time
from src.config import PREDICTIONS_DIR, RUN_DATE

st.set_page_config(layout='wide')
title = 'Anime Score Predictions'
st.markdown(f"<h1 style='color:#A6483F;text-align:center;'>🌸{title}🌸</h1>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📅 Predictions ", "📊 Statistics"])

with tab1:
    st.caption(
    """
    Predictions are shown for the Top 50 most popular currently airing anime and the Top 50 most anticipated upcoming anime.
        """
    )
    st.caption(
        """
        The accuracy will continue to improve over time.
        """
    )
    st.caption(
        """
        The site is updated monthly, so check back for the latest results!
        """
    )
    st.markdown(
        f"""
        
        <div style='text-align: left;'><span>Current Score: </span>
            <span style='
                background-color:#9ABF15;
                border-radius:16px;
                margin-left:8px;
                min-width:40px;
                min-height:20px;
                display:inline-block;
            '></span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        f"""
        
        <div style='text-align: left;'><span>Prediction Score: </span>
            <span style='
                background-color:#5368A6;
                margin-left:8px;
                border-radius:16px;
                min-width:40px;
                min-height:20px;
                display:inline-block;
            '></span>
        </div>
        """,
        unsafe_allow_html=True
    )

with tab2:
    st.write("Coming soon...")

col1, _, col2 = st.columns([1, 0.1, 1])


def get_api_response(url: str) -> requests.Response | None:
    while True:
        response = requests.get(url)
        if response.status_code == 200:
            return response
        elif response.status_code == 429:
            print('Rate limit exceeded. Waiting for 1 second...')
            time.sleep(1)
        else:
            print(f'Error {response.status_code}: {response.text}')
            return None

@st.cache_resource
def fetch_anime_info(mal_id):
    url = f"https://api.jikan.moe/v4/anime/{mal_id}"
    while True:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json().get("data", [])
            return {
                'title_english': data.get('title_english'),
                'title': data.get('title', 'Unkown'),
                'score': data.get('score', 'N/A'),
                'image_url': data.get("images", {}).get("jpg", {}).get("image_url", "")
            }
        elif response.status_code == 429:
            print('Rate limit exceeded. Waiting for 1 second...')
            time.sleep(1)
        else:
            print(f'Error {response.status_code}: {response.text}')
            return None    

    
def show_airing(airing):
    columns_per_row = 5
    with col1:
        st.markdown("<h1 style='color:#63CAF2;text-align:center;'>Currently Airing</h1>", unsafe_allow_html=True)
        for i in range(0, len(airing), columns_per_row):
            cols = st.columns(columns_per_row)
            for idx, row in enumerate(airing.iloc[i:i + columns_per_row].itertuples(index=False)):
                with cols[idx]:
                    mal_id = int(row.mal_id)
                    prediction = row.predicted_score

                    data = fetch_anime_info(mal_id)
                    title_english = data.get('title_english')
                    title = data.get("title")
                    title = title_english if title_english else title
                    score = data.get("score")
                    image_url = data.get('image_url')

                    st.image(image_url, caption=title)
                    formatted_score = f"{score:.2f}⭐" if score is not None else "N/A"
                    st.markdown(
                        f"""
                        <div style='text-align: center;'>
                            <span style='
                                background-color:#9ABF15;
                                color:white;
                                padding:2px 8px 2px 8px;
                                border-radius:16px;
                                font-size:16px;
                                margin-left:8px;
                                margin-bottom: 8px;
                                display:inline-block;
                            '>{formatted_score}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"""
                        <div style='text-align: center;'>
                            <span style='
                                background-color:#5368A6;
                                color:white;
                                padding:2px 8px 2px 8px;
                                border-radius:16px;
                                font-size:16px;
                                margin-left:8px;
                                display:inline-block;
                            '>{prediction:.2f}⭐</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                time.sleep(0.6)
            st.divider()


def show_unreleased(unreleased):
    columns_per_row = 5
    years = sorted(unreleased['year'].unique().tolist())

    with col2:
        for year in years:
            st.markdown(f"<h1 style='color:#63CAF2;text-align:center;'>{year} Unreleased Anime</h1>", unsafe_allow_html=True)

            year_anime = unreleased[unreleased['year'] == year]

            for i in range(0, len(year_anime), columns_per_row):
                cols = st.columns(columns_per_row)
                for idx, row in enumerate(year_anime.iloc[i:i + columns_per_row].itertuples(index=False)):
                    with cols[idx]:
                        mal_id = int(row.mal_id)
                        prediction = row.predicted_score

                        data = fetch_anime_info(mal_id)
                        title_english = data.get('title_english')
                        title = data.get("title")
                        title = title_english if title_english else title
                        image_url = data.get('image_url')

                        st.image(image_url, caption=title)
                        st.markdown(
                            f"""
                            <div style='text-align: center;'>
                                <span style='
                                    background-color:#5368A6;
                                    color:white;
                                    padding:2px 8px 2px 8px;
                                    border-radius:16px;
                                    font-size:16px;
                                    margin-left:8px;
                                    display:inline-block;
                                '>{prediction:.2f}⭐</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    time.sleep(0.6)
                st.divider()

@st.cache_data
def load_predictions():
    airing = pd.read_csv(f'{PREDICTIONS_DIR}/{RUN_DATE}/predictions_airing.csv')
    unreleased = pd.read_csv(f'{PREDICTIONS_DIR}/{RUN_DATE}/predictions_unreleased.csv')
    return airing, unreleased


def main():
    # setup_page()
    airing, unreleased = load_predictions()
    show_airing(airing)
    show_unreleased(unreleased)

main()