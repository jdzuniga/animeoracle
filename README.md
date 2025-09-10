# Anime Oracle

## Project Overview
This project predicts anime scores using historical data and features such as producers, studios, genres, and more. The goal is to provide anime enthusiasts with insights into the potential popularity of currently airing and upcoming shows. While still in its early stages, the project lays the foundation for more advanced improvements in the future.

---

## Website
[![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://animeoracle.azurewebsites.net/)
[![Streamlit App](https://img.shields.io/badge/Website-Live-green?style=for-the-badge&logo=streamlit)](https://animeoracle.azurewebsites.net/)
- Top 50 Popular Anime with predicted scores
- Unreleased Anime Predictions by release year

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
- Type, Source, Rating, Trailer indicator, Sequel indicator, Season (one-hot encoding)
- Year of release

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

Dimensionality Reduction: TruncatedSVD for sparse features

Modeling: LightGBM regressor tuned for root mean squared error (RMSE)

## Performance
- Baseline heuristic MAE: ~0.70
- LightGBM
  - MAE: ~0.40
  - RMSE: ~0.50
  - R²: ~0.50

## Future Improvements
- Implement more robust feature selection to improve model accuracy  
- Ensemble models for better performance  
- Improve offline testing with larger validation windows and stricter evaluation  
- Add weekly batch training and prediction for near real-time updates  

## Author
**David Zuniga**  


