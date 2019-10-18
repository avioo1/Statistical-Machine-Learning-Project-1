import pandas as pd
from collections import Counter
import operator
file_path = './data/train_tweets_mod.csv'

dfs = pd.read_csv(file_path)
num_entries = 0
users = dfs['User'].values.tolist()

user_tweet_dict = Counter(users)

sorted_user_tweet_dict = sorted(user_tweet_dict.items(), key=operator.itemgetter(1), reverse=True)

values = user_tweet_dict.values()

count_dict = Counter(values)

sorted_count_dict = sorted(count_dict.items(), key=operator.itemgetter(1), reverse=True)

valid_users = [user[0] for user in sorted_user_tweet_dict if user[1] > 25]