import numpy as np
import pandas as pd

# reading the dataset
prod_df = pd.read_csv("hairproducts.csv")

print(prod_df.head())
# print(prod_df.shape)
# print(prod_df.info())

# print(prod_df.head(1)['Use'])


from sklearn.feature_extraction.text import TfidfVectorizer

# removing the unnecessary words from the vectorizer
tfv = TfidfVectorizer(min_df=3, max_features=None,
        strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
        ngram_range=(1,3),
        stop_words = 'english')

# Filling NaNs with empty string
prod_df['Use'] = prod_df['Use'].fillna('')

# creating the sparse matrix
tfv_matrix = tfv.fit_transform(prod_df['Use'])

# print(tfv_matrix)
# print(tfv_matrix.shape)

# computing the sigmoid kernel
from sklearn.metrics.pairwise import sigmoid_kernel
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

# print(sig[0])

# reverse mapping of indices and movie titles
indices = pd.Series(prod_df.index, index=prod_df['Product']).drop_duplicates()
# print(indices)


# giving recommendation
def give_rec(title, sig=sig):
    # get the index corresponding to 'Product'
    idx = indices[title]

    # get the pairwise similarity scores
    sig_scores = list(enumerate(sig[idx]))

    # sorting the products
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)  # sorting it in the ascending order

    # Scores of the 3 most similar products
    sig_scores = sig_scores[1:4]

    # product indices
    prod_indices = [i[0] for i in sig_scores]

    # Top 3 similar products
    return prod_df['Product'].iloc[prod_indices]

rec = give_rec('Shaving cream')
print(rec)










