import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from predictor import valid_users
from gensim.models import KeyedVectors
from utilities import get_vector
from ClassifierClass import ClassifierClass
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

file_path = './data/finaltrain2.csv'
w2v_path = './data/w2v_model_M.bin'
# w2v_path = './data/w300sm.bin'

test_df = pd.read_csv("./data/finaltest2.csv")
X_test = [x.lower() if str(x) != "nan" else "" for x in test_df['Clean Tweets'].tolist()]

df_main = pd.read_csv(file_path)
w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
# w2v_model = KeyedVectors.load(w2v_path)
sid = SentimentIntensityAnalyzer()

y = df_main['User'].values.tolist()
X0 = [x.lower() if str(x) != "nan" else "" for x in df_main['Clean Tweets'].tolist()]


Zipped_xy = [[*x] for x in zip(*[(xx, yy) for xx, yy in zip(X0, y) if int(yy) in valid_users])]

y = np.array(Zipped_xy[1])
cv = CountVectorizer(max_features=3696, ngram_range=(1, 2))
cv.fit(Zipped_xy[0] + X_test)
# w2v
X1 = np.array([get_vector(x, w2v_model) for x in Zipped_xy[0]])
# Adding Sentiment
X2 = np.array([list(sid.polarity_scores(x).values()) if str(x) != "nan" else [0, 0, 0, 0] for x in Zipped_xy[0]])
# CV
X3 = cv.transform(Zipped_xy[0]).toarray()

X = np.array([np.append(x1, x2, axis=0) for x1, x2 in zip(X1, X2)])
X = np.array([np.append(x1, x2, axis=0) for x1, x2 in zip(X, X3)])
print(X.shape, y.shape)

"""
TRAIN
"""

X, y = shuffle(X, y, random_state=0)
CLF = ClassifierClass(vm=w2v_model, vm2=cv)
CLF.fit(X, y, run_cross_val=False)
CLF.save("./models/model_v3.p")

"""
TEST
"""


# w2v
X1 = np.array([get_vector(tweet, w2v_model) for tweet in test_df['Clean Tweets']])
# Adding Sentiment
X2 = np.array([list(sid.polarity_scores(x).values()) if str(x) != "nan" else [0, 0, 0, 0] for x in test_df['Clean Tweets']])
# CV
X3 = cv.transform(X_test).toarray()

X_test = np.array([np.append(x1, x2, axis=0) for x1, x2 in zip(X1, X2)])
X_test = np.array([np.append(x1, x2, axis=0) for x1, x2 in zip(X_test, X3)])

y_preds = CLF.predict(X_test)
print(y_preds)
answer_df = pd.DataFrame({'Id': range(1, len(y_preds) + 1), 'Predicted': y_preds})

answer_df.to_csv("outputs/submission3.csv", index=None)

tp = 1
