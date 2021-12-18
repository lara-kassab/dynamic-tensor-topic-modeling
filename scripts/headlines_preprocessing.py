import numpy as np
import nltk
import pickle
import pandas as pd
import tensorly as tl
from tensorly import fold
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

def generate_tensor(directory, if_save=False):
    """
    This code generates the TFIDF tensor for the 1M Headlines dataset.
    Args:
        directory (srt): pathname to the ABC news csv data file
    Return:
        headlines_tensor (ndarray): 1M Headlines TFIDF tensor with dimensions: (time, vocabulary, headlines)
    """
    np.random.seed(0)

    # Load ABC news dataset
    data = pd.read_csv(directory, error_bad_lines=False, warn_bad_lines = True);

    # Format the date of publishing
    data['publish_date'] = pd.to_datetime(data['publish_date'],format='%Y%m%d')

    ## Randomly subsample dataset
    # Subsample dataset per month
    data_subsample = data.copy()
    sample_val = 700 # number of headlines per month
    sub_indices = data['publish_date'].groupby([data.publish_date.dt.year, data.publish_date.dt.month]).sample(n=sample_val, replace = False).index
    data_subsample = data_subsample.iloc[sub_indices]

    # Sort data by chronological order
    data_subsample.sort_values(by=['publish_date'], inplace=True)

    # Create a yyyy-mm column
    data_subsample['month_year'] = pd.to_datetime(data_subsample['publish_date']).dt.to_period('M')

    # Convert the headline text in the dataframe into a corpus (list)
    corpus = data_subsample['headline_text'].values.tolist()

    ## Extract TF-IDF weights and features
    n_features = 7000
    max_df=0.7
    min_df=5
    stop_words_list = nltk.corpus.stopwords.words('english')
    stop_words_list.append("abc") # remove the common string "abc" which could stand for "Australian Broadcasting Corporation"
    vectorizer = TfidfVectorizer(max_df=max_df,
                                    min_df=min_df,
                                    max_features=n_features,
                                    stop_words=stop_words_list)
    vectors = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = np.array(dense).transpose()

    # Organize data into a third-order tensor
    n_div = 1 # number of months per time slice
    n_headlines = sample_val*n_div # number of headlines per time slice
    headlines_tensor = [feature_names, fold(denselist, 1, (denselist.shape[1] // n_headlines, denselist.shape[0], n_headlines))]

    print('!!! headlines_tensor[1].shape', headlines_tensor[1].shape)
    print('!!! headlines_tensor[0].shape', headlines_tensor[0][:10])


    print("Shape of the tensor {}.".format(headlines_tensor[1].shape))
    print("Dimensions: (time, vocabulary, headlines)")

    if if_save:
        path = "Data/headlines_tensor.pickle"
        with open(path, 'wb') as handle:
                pickle.dump(headlines_tensor, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return headlines_tensor
