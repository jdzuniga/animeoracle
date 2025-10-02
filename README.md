# Anime Oracle
## Business Problem
The anime industry is highly competitive, with studios, streaming platforms, and advertisers constantly seeking to identify shows that will capture audience attention and generate revenue. Investing in the wrong title can lead to wasted production and marketing budgets, while a hit series can drive significant profits through subscriptions, merchandise, and advertising.

### The key challenge
How can stakeholders predict and understand which anime are most likely to achieve high audience ratings and popularity, in order to prioritize investment and maximize returns?

## Project Objective
This project leverages machine learning to predict anime scores based on historical data (genres, studios, producers, engineered metrics, etc.) and provides analytical insights into the factors most strongly associated with success. By predicting anime scores and analyzing the features most associated with high performance, this project provides actionable insights into which shows are most likely to generate hype and revenue. The solution not only predicts audience reception but also explains what makes an anime more likely to perform well. While still in its early stages, the project lays the foundation for more advanced improvements in the future.

By reframing the prediction task as a business decision-support tool, stakeholders can:
- Reduce financial risk by identifying low-performing shows before launch. 
- Optimize marketing spend toward titles with high predicted hype. 
- Understand long-term trends in genre, studio performance, and viewer preferences. 
- Make data-driven licensing and production decisions.

## Key Analytical Questions
The dashboard and charts answer critical business questions such as:

1. What are the emerging or declining genre trends in the anime industry?
2. Do certain release windows (Spring, Summer, Fall, Winter) perform better than others ?

### Key Insight: Production Trends
![Genres](https://i.imgur.com/aLHaCn3.png)

Anime production is heavily concentrated in Fantasy and Action, with Comedy and Adventure also remaining strong. While these genres dominate the supply, the market risks oversaturation. Mid-tier genres like Romance and Drama show steady resilience, suggesting opportunities for growth in emotionally-driven storytelling. Projections for 2026 remain incomplete, but early signals suggest continued dominance of Fantasy and Action.
### Key Insight: Seasonality in Anime Ratings
![Seasonality 2024](https://i.imgur.com/hiGgU0X.png)

![Seasonality 2000-2024](https://i.imgur.com/lV7VT6X.png)

Anime released in Spring 2024 performed better on average than those in other quarters.
Studios and platforms shouldn’t over-interpret one strong Spring season as a structural trend. Instead, focus on the titles and genres driving the Q2 2024 spike, since the broader multi-year trend suggests seasonality is usually weak.

---
## Website
[![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://animeoracle.azurewebsites.net/)
[![Streamlit App](https://img.shields.io/badge/Website-Live-green?style=for-the-badge&logo=streamlit)](https://animeoracle.azurewebsites.net/)

The website provides an interactive interface for exploring predictions and insights:

**Top 100 Currently Airing Anime**
* Predicted score (subject to change until series completion).
* Current ratings may fluctuate as more viewers watch and rate the show
* Helps viewers and platforms track which titles are gaining the most traction.

**Top 100 Unreleased Anime**
* Predicts ratings for upcoming shows before they air. 
* Provides studios, streaming platforms, and advertisers with an early indicator of which titles are most likely to generate hype and drive engagement.

---

## Tech Stack
- **Python** (data processing, modeling)
- **Pandas / NumPy** (data manipulation)
- **Scikit-learn / LightGBM** (machine learning models)
- **Jikan API** (anime data retrieval)
- **Streamlit** (web app framework)
- **Azure** (deployment & hosting)

---

## Data & Features
- Source: Scraped from [MyAnimeList](https://myanimelist.net) using the **Jikan API**.
- Title (TF-IDF encoding)
- Studios, Demographics, Genres, Themes, Producers (multi-label encoding)
- Type, Source, Rating, Trailer indicator, Sequel indicator
- Year of release, Month of release

---

## Project Structure
```
├── src/               # Source code
├── main.py            # Entrypoint
├── webapp.py          # Streamlit app
├── requirements.txt   # Dependencies
└── README.md          # Project description
```

## Running the App

```bash
# Clone the repo
git clone https://github.com/jdzuniga/animeoracle.git

# Install dependencies
pip install -r requirements.txt

# Run the whole pipeline
python main.py

# Run the Streamlit app
streamlit run webapp.py
```

## Model & Prediction Pipeline
Preprocessing
- Encoding (TF-IDF, One-hot, Multi-label)
- Imputation for missing values
- Scaling numerical features
- Grouping rare categories


Modeling: LightGBM regressor tuned for root mean squared error (RMSE)

## Performance
- Baseline heuristic MAE: ~0.70
- LightGBM
  - MAE: ~0.40
  - RMSE: ~0.54
  - R²: ~0.45

## Future Improvements
- Implement more robust feature selection to improve model accuracy  
- Ensemble models for better performance  
- Improve offline testing with larger validation windows and stricter evaluation  
- Add weekly batch training and prediction for near real-time updates


