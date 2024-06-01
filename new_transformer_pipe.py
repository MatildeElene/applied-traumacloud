import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfTransformer
import new_transformer_concat as new_transformer

#Adding Robertas pipeline (updated version with tfidf + softmax)
df = pd.read_csv('countvec_030_010.csv')
df = df.set_index('category')

transformer = TfidfTransformer()
idf = transformer.fit(new_transformer.word_counts_by_category.values) #using this approach to reuse the fitted transformer for multiple datasets.
tfidf = idf.transform(df).todense()

soft2 = pd.DataFrame(softmax(tfidf, axis=0), index=df.index, columns=df.columns)

print("Top 20 terms for 'transposed_eating_disorders5':")
print(soft2.loc['eating_disorders'].nlargest(20))

print("\nTop 20 terms for 'transposed_miscarriage5':")
print(soft2.loc['miscarriage'].nlargest(20))

print("\nTop 20 terms for 'transposed_war_trauma5':")
print(soft2.loc['war_trauma'].nlargest(20))

#transpose categories for wordcloud visualisation 
transposed_category_eat = soft2.loc['eating_disorders'].T
transposed_category_eat.to_csv('transposed_eating_disorders5.csv')

transposed_category_mis = soft2.loc['miscarriage'].T
transposed_category_mis.to_csv('transposed_miscarriage5.csv')

transposed_category_war = soft2.loc['war_trauma'].T
transposed_category_war.to_csv('transposed_war_trauma5.csv')