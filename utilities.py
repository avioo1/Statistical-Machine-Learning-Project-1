import re
from nltk.corpus import stopwords
import numpy as np

stop_words = stopwords.words("english")


def cleaner(tweet):
    # removing @usernames and redundant spaces
    pre_mod_tweet = re.sub('@[^\s]+', '', tweet)

    # removing hyperlinks and emojis
    pre_mod_tweet = re.sub(' +', ' ', re.sub('@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+', ' ',
                                             str(pre_mod_tweet).lower())).strip()

    # removing stopwords
    mod_tweet = []
    for token in pre_mod_tweet.split():
        # if token not in stop_words:
            mod_tweet.append(token)
    return " ".join(mod_tweet)


def get_vector(tweet_word_list, w2v_model):
    try:
        valid_words = [word for word in tweet_word_list.split(" ") if word in w2v_model.vocab]
        if not valid_words:
            return np.array([0.0] * w2v_model.vector_size)
        w2v_vector = np.true_divide(np.sum([[w2v_model[word]] for word in valid_words], axis=0), len(valid_words))
        return w2v_vector[0]
    except:
        return np.array([0.0] * w2v_model.vector_size)
