import pandas as pd
from utilities import cleaner
"""
Train Modification
"""
k = 1

with open("./data/given/train_tweets.txt", "r") as rf:
    ct = rf.readlines()
    ct_user = [c.split()[0] for c in ct]
    ct_tweet = [" ".join(c.split()[1:]) for c in ct]

df = pd.DataFrame({"User": ct_user, "Tweets": ct_tweet})

clean_tweets = []
for i in range(len(df)):
    clean_tweets.append(cleaner(df['Tweets'][i]))

pd.DataFrame({"User": ct_user, "Tweets": ct_tweet, "Clean Tweets": clean_tweets}).to_csv("./data/train_tweets_mod2.csv",
                                                                                         index=False)

"""
Test Modification
"""
k = 1

with open("./data/given/test_tweets_unlabeled.txt", "r") as rf:
    ct = rf.readlines()

test_df = pd.DataFrame({"Tweets": ct})

clean_tweets = []
for i in range(len(test_df)):
    clean_tweets.append(cleaner(test_df['Tweets'][i]))

pd.DataFrame({"Clean Tweets": clean_tweets}).to_csv("./data/test_tweets_mod2.csv", index=False)


