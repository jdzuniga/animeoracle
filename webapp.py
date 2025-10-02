from PIL import Image
from pathlib import Path
import streamlit as st
import pandas as pd
import argparse
from src.config import PREDICTIONS_DIR, RUN_DATE, POSTERS_DIR


st.set_page_config(
    page_title="Anime Oracle",
    page_icon="🐻",
    layout="wide"
)

hide_spinner_style = """
<style>
#MainMenu {visibility: hidden;}        /* hides hamburger menu */
footer {visibility: hidden;}          /* hides footer */
header {visibility: hidden;}          /* hides Streamlit header */
[data-testid="stDecoration"] {display: none;} /* hides top-right spinner */
</style>
"""
st.markdown(hide_spinner_style, unsafe_allow_html=True)


title = 'Anime Oracle'
st.markdown(f"<h1 style='font-size: 70px;color:#FFFFFF;text-align:center;'>{title}</h1>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📅 Predictions ", "📊 Statistics"])


@st.cache_resource
def preload_posters(df):
    posters = {}
    root = Path(__file__).resolve().parent
    for row in df.itertuples(index=False):
        file_name = f"{row.mal_id}.webp"
        path = root / POSTERS_DIR / RUN_DATE / file_name
        posters[row.mal_id] = Image.open(path)
    return posters


with tab1:
    st.caption(
        """
        The accuracy will continue to improve over time. The site is updated monthly, so check back for the latest results!
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


def show_airing(airing, posters, col):
    columns_per_row = 4
    title = "Top 100 <span style='color:#E259F5;'>Currently Airing</span>"

    with col:
        st.markdown(
            f"<h1 style='color:white;text-align:center;'>{title}</h1>",
            unsafe_allow_html=True
        )
        (airing_tab,) = st.tabs(["Airing"])
        with airing_tab:
            for i in range(0, len(airing), columns_per_row):
                cols = st.columns(columns_per_row)
                for idx, row in enumerate(airing.iloc[i:i + columns_per_row].itertuples(index=False)):
                    with cols[idx]:
                        title = row.title
                        score = row.score
                        formatted_score = f"{score:.2f}" if score is not None else "N/A"
                        prediction = f'{row.predicted_score:.2f}'

                        poster = posters[row.mal_id]
                        st.image(poster, caption=title)

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


def show_unreleased(unreleased, posters, col):
    columns_per_row = 4
    years = sorted(unreleased['year'].unique().tolist())

    title = "Top 100 <span style='color:#E259F5;'>Unreleased</span>"

    with col:
        st.markdown(
            f"<h1 style='color:white;text-align:center;'>{title}</h1>",
            unsafe_allow_html=True
        )
        tabs = st.tabs([str(year) + ' Forecast' for year in years])
        years_tabs = [*tabs]
        for year, tab in zip(years, years_tabs):
            with tab:
                year_anime = unreleased[unreleased['year'] == year]

                for i in range(0, len(year_anime), columns_per_row):
                    cols = st.columns(columns_per_row)
                    for idx, row in enumerate(year_anime.iloc[i:i + columns_per_row].itertuples(index=False)):
                        with cols[idx]:
                            title = row.title
                            prediction = f'{row.predicted_score:.2f}'

                            poster = posters[row.mal_id]
                            st.image(poster, caption=title)

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


@st.cache_data
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


def display_footer():
    st.markdown(
        f"""
        <hr>
        <p style='text-align: center; font-size: 16px; color: gray;'>
            Last updated: {RUN_DATE} | © 2025 Anime Oracle
        </p>
        """,
        unsafe_allow_html=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date format: YYYY-MM-DD"
    )
    args = parser.parse_args()
    if args.date:
        global RUN_DATE
        RUN_DATE = args.date

    max_title_length = 30
    airing, unreleased = load_predictions()

    airing['title'] = shorten_text(airing['title'], max_title_length)
    unreleased['title'] = shorten_text(unreleased['title'], max_title_length)

    airing_posters = preload_posters(airing)
    unreleased_posters = preload_posters(unreleased)

    columns_per_block = 4  # how many per side per row
    max_rows = max(len(airing), len(unreleased)) // columns_per_block + 1


    col1, _, col2 = st.columns([1, 0.1, 1])
    airing_title = "Top 100 <span style='color:#E259F5;'>Currently Airing</span>"
    unreleased_title = "Top 100 <span style='color:#E259F5;'>Unreleased</span>"

    with col1:
        st.markdown(
            f"<h1 style='color:white;text-align:center;'>{airing_title}</h1>",
            unsafe_allow_html=True
        )
        (airing_tab,) = st.tabs(["Airing"])

    with col2:
        st.markdown(
            f"<h1 style='color:white;text-align:center;'>{unreleased_title}</h1>",
            unsafe_allow_html=True
        )
        years = sorted(unreleased['year'].unique().tolist())
        tabs = st.tabs([str(year) + ' Forecast' for year in years])
        years_tabs = [*tabs]


    for i in range(max_rows):
        with col1:
            with airing_tab:
                block = airing.iloc[i * columns_per_block:(i + 1) * columns_per_block]
                cols = st.columns(columns_per_block)
                for idx, row in enumerate(block.itertuples(index=False)):
                    with cols[idx]:

                        score = row.score
                        formatted_score = f"{score:.2f}" if not pd.isna(score) else "N/A"
                        prediction = f"{row.predicted_score:.2f}"

                        poster = airing_posters[row.mal_id]
                        st.image(poster, caption=row.title)
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
                if i < max_rows - 2 and not block.empty:
                    st.divider()
                    st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)


        with col2:
            for year, tab in zip(years, years_tabs):
                with tab:
                    year_anime = unreleased[unreleased['year'] == year]
                    block = year_anime.iloc[i * columns_per_block:(i + 1) * columns_per_block]
                    cols = st.columns(columns_per_block)
                    for idx, row in enumerate(block.itertuples(index=False)):
                        with cols[idx]:
                            poster = unreleased_posters[row.mal_id]
                            st.image(poster, caption=row.title)

                            prediction = f"{row.predicted_score:.2f}"
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
                    if i < max_rows - 2 and not block.empty:
                        st.divider()
                        st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
    display_footer()


main()