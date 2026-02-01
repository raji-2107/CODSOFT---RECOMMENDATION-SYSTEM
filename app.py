from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

data = {
    'User': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'],
    'Movie': ['Avengers', 'Titanic', 'Inception',
              'Avengers', 'Inception',
              'Titanic', 'Avatar',
              'Avengers', 'Avatar'],
    'Rating': [5, 4, 5, 4, 5, 5, 4, 3, 5]
}

df = pd.DataFrame(data)

user_movie_matrix = df.pivot_table(
    index='User',
    columns='Movie',
    values='Rating'
).fillna(0)

similarity_matrix = cosine_similarity(user_movie_matrix)
similarity_df = pd.DataFrame(
    similarity_matrix,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)

def recommend_movies(user):
    similar_users = similarity_df[user].sort_values(ascending=False)[1:]
    watched = user_movie_matrix.loc[user]
    watched_movies = watched[watched > 0].index.tolist()

    recommendations = {}

    for sim_user in similar_users.index:
        similarity_score = similarity_df.loc[user, sim_user]
        for movie in user_movie_matrix.columns:
            if movie not in watched_movies:
                rating = user_movie_matrix.loc[sim_user, movie]
                if rating > 0:
                    recommendations[movie] = recommendations.get(movie, 0) + rating * similarity_score

    return sorted(recommendations.keys())

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    user = ""

    if request.method == 'POST':
        user = request.form['user']
        if user in user_movie_matrix.index:
            recommendations = recommend_movies(user)

    return render_template('index.html', recommendations=recommendations, user=user)

if __name__ == '__main__':
    app.run(debug=True)
