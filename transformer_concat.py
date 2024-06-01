import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfTransformer

# File paths for each CSV file +  Initialize an empty list to store data + creating category names
file_paths = ['traumas/eating_disorder_training.csv', 'traumas/miscarriage_training.csv', 'traumas/war_trauma_training.csv']
data = []
new_category_names = ['eating_disorders', 'miscarriage', 'war_trauma']

for file_path, category_name in zip(file_paths, new_category_names):
    df = pd.read_csv(file_path, usecols=['clean_text'])
    df['category'] = category_name  
    data.append(df)

# Concatenating the DataFrames into a single DataFrame
combined_df = pd.concat(data, ignore_index=True)

# Count word frequencies for each category
count_vectorizer = CountVectorizer(max_df=0.30, min_df=0.10) 
word_counts = count_vectorizer.fit_transform(combined_df['clean_text'])

# Convert word counts matrix to DataFrame
word_counts_df = pd.DataFrame(word_counts.toarray(), columns=count_vectorizer.get_feature_names_out())

# Extra filtering
word_counts_df = word_counts_df.loc[:, word_counts_df.columns.str.contains(r'^[a-zA-Z]')]
7

# Add category column to word counts DataFrame. group by category and sum the word counts
word_counts_df['category'] = combined_df['category']
word_counts_by_category = word_counts_df.groupby('category').sum()

word_counts_by_category.to_csv('countvec_030_010.csv')
