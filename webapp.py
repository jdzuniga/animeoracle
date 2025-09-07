# from PIL import Image
from pathlib import Path
import streamlit as st
import pandas as pd
from src.config import PREDICTIONS_DIR, RUN_DATE, POSTERS_DIR

st.set_page_config(layout='wide')
title = 'Anime Score Predictions'
st.markdown(f"<h1 style='font-size: 60px;color:#F53B5C;text-align:center;'>🌸{title}🌸</h1>", unsafe_allow_html=True)

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


def show_airing(airing):
    columns_per_row = 4
    with col1:
        st.markdown("<h1 style='color:#63CAF2;text-align:center;'>Currently Airing</h1>", unsafe_allow_html=True)
        (airing_tab,) = st.tabs(["Airing"])
        with airing_tab:
            for i in range(0, len(airing), columns_per_row):
                cols = st.columns(columns_per_row)
                for idx, row in enumerate(airing.iloc[i:i + columns_per_row].itertuples(index=False)):
                    with cols[idx]:
                        mal_id = row.mal_id
                        title = row.title
                        score = row.score
                        formatted_score = f"{score:.2f}" if score is not None else "N/A"
                        prediction = f'{row.predicted_score:.2f}'

                        st.image(f'{POSTERS_DIR}/{RUN_DATE}/{mal_id}.webp', caption=title)

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
                                '>{formatted_score}⭐</span>
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
                                '>{prediction}⭐</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                st.divider()
                st.caption("\n\n\n\n")  


def show_unreleased(unreleased):
    columns_per_row = 4
    years = sorted(unreleased['year'].unique().tolist())

    with col2:
        st.markdown(f"<h1 style='color:#F5ED5D;text-align:center;'>Unreleased Anime</h1>", unsafe_allow_html=True)
        tabs = st.tabs([str(year) + ' Forecast' for year in years])
        years_tabs = [*tabs]
        for year, tab in zip(years, years_tabs):
            with tab:
                year_anime = unreleased[unreleased['year'] == year]

                for i in range(0, len(year_anime), columns_per_row):
                    cols = st.columns(columns_per_row)
                    for idx, row in enumerate(year_anime.iloc[i:i + columns_per_row].itertuples(index=False)):
                        with cols[idx]:
                            mal_id = row.mal_id
                            title = row.title
                            prediction = f'{row.predicted_score:.2f}'

                            st.image(f'{POSTERS_DIR}/{RUN_DATE}/{mal_id}.webp', caption=title)

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
                                        margin-bottom: 8px;
                                        display:inline-block;
                                    '>{prediction}⭐</span>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            st.markdown(
                            f"""
                            <div style='text-align: center;'>
                                <span style='
                                    background-color:#00000;
                                    color:black;
                                    padding:2px 8px 2px 8px;
                                    border-radius:16px;
                                    font-size:16px;
                                    margin-left:8px;
                                    display:inline-block;
                                '>&nbsp;</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    st.divider()
                    st.caption("\n\n\n\n")  


def shorten_text(series, length):
    return series.apply(lambda text: text if len(text) < length else text[:length] + '...')


@st.cache_data
def load_predictions():
    root = Path(__file__).resolve().parent
    airing_path = root / PREDICTIONS_DIR / RUN_DATE / 'predictions_airing.csv'
    unreleased_path = root / PREDICTIONS_DIR / RUN_DATE / 'predictions_unreleased.csv'

    airing = pd.read_csv(airing_path)
    unreleased = pd.read_csv(unreleased_path)

    return airing, unreleased


def main():
    # setup_page()
    airing, unreleased = load_predictions()

    max_title_length = 30
    airing['title'] = shorten_text(airing['title'], max_title_length)
    unreleased['title'] = shorten_text(unreleased['title'], max_title_length)

    show_airing(airing)
    show_unreleased(unreleased)

main()