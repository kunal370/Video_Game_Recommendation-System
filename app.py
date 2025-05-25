import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity


# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('games.csv')
    df['game_mode'] = df['game_mode'].str.split(', ')
    df['language'] = df['language'].str.split(', ')
    return df


df = load_data()

# Initialize encoders
mlb_game_mode = MultiLabelBinarizer()
mlb_language = MultiLabelBinarizer()
ohe_genre = OneHotEncoder(sparse_output=False)

# Fit encoders
game_mode_encoded = mlb_game_mode.fit_transform(df['game_mode'])
language_encoded = mlb_language.fit_transform(df['language'])
genre_encoded = ohe_genre.fit_transform(df[['genre']])

# Create feature matrix
feature_matrix = pd.concat([
    pd.DataFrame(game_mode_encoded, columns=mlb_game_mode.classes_),
    pd.DataFrame(language_encoded, columns=mlb_language.classes_),
    pd.DataFrame(genre_encoded, columns=ohe_genre.categories_[0]),
], axis=1)

# Streamlit UI
st.title('🎮 Video Game Recommendation System')

# User inputs
selected_modes = st.multiselect(
    'Select game modes:',
    options=mlb_game_mode.classes_,
)
selected_langs = st.multiselect(
    'Select languages:',
    options=mlb_language.classes_,
)
selected_genres = st.multiselect(
    'Select genres:',
    options=ohe_genre.categories_[0],
)
min_rating = st.slider(
    'Minimum rating:',
    min_value=0.0,
    max_value=5.0,
    value=4.0,
    step=0.1
)


# Encode user input
def encode_user_input(modes, langs, genres):
    # Game modes
    mode_vector = mlb_game_mode.transform([modes])
    # Languages
    lang_vector = mlb_language.transform([langs])
    # Genres (multi-hot)
    if genres:
        genre_df = pd.DataFrame({'genre': genres})
        genre_part = ohe_genre.transform(genre_df).sum(axis=0, keepdims=True)
    else:
        genre_part = np.zeros((1, len(ohe_genre.categories_[0])))
    # Combine
    user_vector = np.hstack([mode_vector, lang_vector, genre_part])
    return user_vector


if st.button('Recommend Games'):
    if not (selected_modes or selected_langs or selected_genres):
        st.warning("Please select at least one preference.")
    else:
        # Encode user input
        user_vector = encode_user_input(selected_modes, selected_langs, selected_genres)

        # Compute similarity
        similarities = cosine_similarity(user_vector, feature_matrix)

        # Filter and sort
        games_with_scores = [
            (i, similarities[0][i], df.iloc[i]['rating'])
            for i in range(len(df))
            if df.iloc[i]['rating'] >= min_rating
        ]
        # Sort by similarity (desc) and rating (desc)
        games_with_scores.sort(key=lambda x: (-x[1], -x[2]))

        # Display results
        if not games_with_scores:
            st.error("No games match your criteria.")
        else:
            st.subheader("🎮 Recommended Games")
            for i, (idx, sim, rating) in enumerate(games_with_scores[:5], 1):
                game = df.iloc[idx]
                st.write(f"{i}. **{game['game_name']}**")
                st.write(f"   - Modes: {', '.join(game['game_mode'])}")
                st.write(f"   - Languages: {', '.join(game['language'])}")
                st.write(f"   - Genre: {game['genre']}")
                st.write(f"   - Rating: ⭐ {game['rating']}")
                st.write("---")