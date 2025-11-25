import streamlit as st
import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report



BESTSELLER_RF_ACCURACY = 0.84  
GENRE_RF_ACCURACY = 0.60
GENRE_REPORT = """
Genre Classification Report:
              precision    recall  f1-score   support

           0       0.75      0.75      0.75         4
           1       0.00      0.00      0.00         1

    accuracy                           0.60         5
   macro avg       0.38      0.38      0.38         5
weighted avg       0.60      0.60      0.60         5
"""

# Feature Engineering & Model Setup 

@st.cache_resource
def load_and_train_model():
    """Loads data, engineers features, trains the Random Forest Bestseller model, and returns encoders."""
    try:
        df = pd.read_csv('books_dataset.csv')
    except FileNotFoundError:
        st.error("Error: 'books_dataset.csv' not found. Please ensure the file is in the same directory.")
        return None, None, None

    # Cleaning/Renaming
    df.rename(columns={"User Rating": "User_Rating"}, inplace=True)
    df.loc[df.Author == 'J. K. Rowling', 'Author'] = 'J.K. Rowling'

    # Feature Engineering Functions
    punct = string.punctuation
    def punctuation_percent(text):
        count = sum(1 for ch in text if ch in punct)
        length = len(text) - text.count(" ")
        return round((count / length) * 100, 3) if length > 0 else 0.0

    # Engineered Features
    df["title_len"] = df["Name"].apply(lambda x: len(x.replace(" ", "")))
    df["word_count"] = df["Name"].apply(lambda x: len(x.split()))
    df["punctuation_percent"] = df["Name"].apply(punctuation_percent)

    # Encoding
    genre_encoder = LabelEncoder()
    author_encoder = LabelEncoder()
    df["Genre_Encoded"] = genre_encoder.fit_transform(df["Genre"])
    df["Author_Encoded"] = author_encoder.fit_transform(df["Author"])

    # Define Bestseller Target
    median_reviews = df["Reviews"].median()
    df["Is_Bestseller"] = (
        (df["User_Rating"] >= 4.5) &
        (df["Reviews"] >= median_reviews)
    ).astype(int)

    new_features = [
        "Price", "Reviews", "User_Rating", "Year", "Genre_Encoded", "Author_Encoded",
        "title_len", "word_count", "punctuation_percent"
    ]

    X = df[new_features]
    y = df["Is_Bestseller"]

    # Train Model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    rf_model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
    rf_model.fit(X_train, y_train)

    return rf_model, genre_encoder, author_encoder

#Prediction Function

def predict_bestseller_for_app(model, genre_encoder, author_encoder, name, price, reviews, rating, year, genre, author):
    """Generates features and predicts bestseller status using the trained model."""
    
    punct = string.punctuation
    def punctuation_percent(text):
        count = sum(1 for ch in text if ch in punct)
        length = len(text) - text.count(" ")
        return round((count / length) * 100, 3) if length > 0 else 0.0

    title_len = len(name.replace(" ", ""))
    word_count = len(name.split())
    punct_per = punctuation_percent(name)

    # Encoding new categorical data 
    try:
        genre_val = genre_encoder.transform([genre])[0]
    except ValueError:
        genre_val = 0 
    
    try:
        author_val = author_encoder.transform([author])[0]
    except ValueError:
        author_val = 0 

    sample = np.array([[
        price, reviews, rating, year,
        genre_val, author_val,
        title_len, word_count, punct_per
    ]])

    prob = model.predict_proba(sample)[0][1]
    
    return prob

# Streamlit Application UI 

st.title("📚 Amazon Bestseller Predictor")

# Load model and encoders
model, genre_encoder, author_encoder = load_and_train_model()

if model is not None:
    
    # Create tabs for Model and Analysis
    tab_predict, tab_stats = st.tabs(["🎯 Bestseller Prediction", "📈 Data & Model Analysis"])
    
    with tab_predict:
        st.markdown("---")
        st.sidebar.header("Book Features")
        
        # 1. Collect user inputs (in sidebar)
        name = st.sidebar.text_input("Book Name", "The Next Big Hit Novel")
        author = st.sidebar.text_input("Author Name", "New Author")
        genre_options = list(genre_encoder.classes_) if genre_encoder else ["Fiction", "Non Fiction"]
        genre = st.sidebar.selectbox("Genre", genre_options)
        
        st.sidebar.markdown("---")
        
        price = st.sidebar.number_input("Price ($)", min_value=0.0, value=14.99, step=0.01)
        reviews = st.sidebar.number_input("Reviews", min_value=0, value=15000, step=100)
        rating = st.sidebar.slider("User Rating (0.0 to 5.0)", min_value=0.0, max_value=5.0, value=4.5, step=0.1)
        year = st.sidebar.number_input("Publication Year", min_value=2000, value=2024, step=1)
        
        # 2. Prediction on button click 
        if st.button("Predict Bestseller Status"):
            
            probability = predict_bestseller_for_app(
                model, genre_encoder, author_encoder, name, price, reviews, rating, year, genre, author
            )
            
            confidence = round(probability * 100, 2)
            
            # 3. Display results
            if confidence >= 55.0:
                st.success(f"**Prediction: LIKELY Bestseller!** 🏆")
            else:
                st.warning(f"**Prediction: Not Likely.** 📉")

            st.metric("Confidence", f"{confidence}%")
            st.info("The Bestseller prediction is generated by a **Random Forest Classifier** trained on historical book data, weighing factors like **Reviews**, **User Rating**, and **Price** to calculate the likelihood of success")


    with tab_stats:
        st.header("Statistical & Model Analysis")
        st.markdown("These findings were generated from your `books_dataset.csv` file.")

        st.subheader("Quantitative Model Performance")
        st.markdown("Metrics from the original analysis script:")
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.metric("Bestseller RF Accuracy", f"{BESTSELLER_RF_ACCURACY*100:.2f}%")
            st.code("Classification Report not shown for brevity.")

        with col_m2:
            st.metric("Genre RF Accuracy", f"{GENRE_RF_ACCURACY*100:.2f}%")
            st.code(GENRE_REPORT)


        st.subheader("Bestseller Prediction Model Insights")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image('bestseller_feature_importance.png', caption='Top Factors That Predict a Bestseller (Random Forest)')
        with col2:
            st.image('correlation_heatmap.png', caption='Correlation Heatmap of Features')
            st.markdown("---")
            st.image('genre_feature_importance.png', caption='Feature Importance for Genre Classification')


        st.subheader("Exploratory Data Analysis (EDA)")
        col3, col4 = st.columns(2)
        with col3:
            st.image('genre_distribution_pie_chart.png', caption='Overall Genre Distribution')
            st.image('yearly_genre_distribution.png', caption='Genre Distribution Over Years')
        with col4:
            st.image('price_vs_reviews_scatter.png', caption='Price vs Reviews (Popularity)')
            st.image('author_appearances_bar_chart.png', caption='Top Author Appearances')
