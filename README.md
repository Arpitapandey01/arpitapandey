import networkx as nx
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load your dataset (Assuming you have a CSV file with columns 'user', 'product', 'rating', 'review_text', 'timestamp', 'label')
df = pd.read_csv('your_dataset.csv')

# Create a graph for user-product relationships
G = nx.from_pandas_edgelist(df, 'user', 'product')

# Feature engineering
# Calculate user degree centrality
user_degree_centrality = nx.degree_centrality(G)

# Calculate product degree centrality
product_degree_centrality = nx.degree_centrality(G)

# Combine features into a DataFrame
features = pd.DataFrame({
    'user_degree_centrality': user_degree_centrality,
    'product_degree_centrality': product_degree_centrality,
    # Add more features based on your dataset and analysis
})

# Perform sentiment analysis on review_text
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['review_text'].astype(str))

# Combine sentiment analysis features with existing features
sentiment_features = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
all_features = pd.concat([features, sentiment_features], axis=1)

# Define the target variable
target = df['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_features, target, test_size=0.2, random_state=42)

# Create a machine learning pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# If needed, you can now use the trained model to predict the labels for new data
# new_data = ... # Load or generate new data with the same features
# new_predictions = pipeline.predict(new_data)
