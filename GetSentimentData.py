from numpy import empty
import snscrape.modules.twitter as sntwitter
import pandas as pd
import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging
import re

# sentiment analyzer
analyser = SentimentIntensityAnalyzer()
def sentiment_analyzer_scores(text):
    score = analyser.polarity_scores(text)
    lb = score['compound']
    if lb >= 0.05:
        return 1
    elif (lb > -0.05) and (lb < 0.05):
        return 0
    else:
        return -1


def get_most_liked_retweeted_tweets_for_a_range_of_days(start_date=datetime.datetime(2015, 1, 1), numdays=18455, top_n_tweets=100, save_to_csv_file_name='btc_tweets.csv'):
    '''
        This method is using snscrape to scrape the latest 10000 tweets from Tweetet. 
        It then get the top_n_tweets most liked & retweeted tweets and save them into a csv file.
        Because VADER doens't work with non-English languages. All the rows with lang!=en has being removed from the dataframe. 
        Using translation web api could be a solution for enabling VADER to work on non-english words. But the accuracy will depends on the translation machine and vader

        start_date:             date range start date
        numdays:                number of days from start_date
        top_n_tweets:           top n number of most liked & retweeted tweets
        save_to_csv_file_name:  csv file name to save the top n number of most liked & retweeted tweets
    '''

    # Creating an empty df
    tweets_df = pd.DataFrame([], columns=['Datetime', 'Tweet Id', 'Text', 'Username', 'likeCount', 'retweetCount', 'lang'])
    tweets_df = tweets_df.set_index(['likeCount', 'retweetCount'])
    tweets_df = tweets_df.sort_index(ascending=False)
    tweets_df.to_csv('./Data/{}'.format(save_to_csv_file_name))

    # Creating list to append tweet data to
    tweets_list = []
    dateList = [start_date + datetime.timedelta(days=x) for x in range(numdays)]
    numberOfTweetsToRetrieve = 100

    for date in dateList:
        logging.info('Getting tweets for date {}'.format(date))
        # Using TwitterSearchScraper to scrape data and append tweets to list
        requestString = 'Bitcoin since:{} until:{}'.format(date.strftime("%Y-%m-%d"), (date + datetime.timedelta(days=1)).strftime("%Y-%m-%d"))
        response = sntwitter.TwitterSearchScraper(requestString).get_items()
        for i,tweet in enumerate(response):
            if i % 1000 == 0 and i / 1000 >=1:
                logging.info('{} tweets got'.format(i))
            if i>numberOfTweetsToRetrieve:
                break
            else:
                tweets_list.append([tweet.date, tweet.id, tweet.content, tweet.user.username, tweet.likeCount, tweet.retweetCount, tweet.lang])

        # Creating a dataframe from the tweets list above
        tweets_df_child = pd.DataFrame(tweets_list, columns=['Datetime', 'Tweet Id', 'Text', 'Username', 'likeCount', 'retweetCount', 'lang'])
        tweets_df_child = tweets_df_child[tweets_df_child.lang == 'en']
        tweets_df_child = tweets_df_child.set_index(['likeCount', 'retweetCount'])
        tweets_df_child = tweets_df_child.sort_index(ascending=False)
        tweets_df_child = tweets_df_child[:top_n_tweets]
        logging.info('df create for date {}. df len is {}'.format(date, len(tweets_df_child)))

        with open('./Data/{}'.format(save_to_csv_file_name), 'a') as f:
            tweets_df_child.to_csv(f, header=False)
        
# get_most_liked_retweeted_tweets_for_a_range_of_days(numdays=2, top_n_tweets=10, start_date=datetime.datetime(2021, 1, 1))


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)        
    return input_txt
    
def clean_tweets(tweet):
    # remove twitter Return handles (RT @xxx:)
    tweet = remove_pattern(tweet, "RT @[\w]*:")
    # remove twitter handles (@xxx)
    tweet = remove_pattern(tweet, "@[\w]*")
    # remove URL links (httpxxx)
    tweet = remove_pattern(tweet, "https?://[A-Za-z0-9./]*")
    # remove special characters, numbers, punctuations (except for #)
    tweet = re.sub('^[a-zA-Z ]+$', '', tweet)
    return tweet

def adding_sentiment_score(csv_path='./Data/btc_tweets.csv'):
    tweets_df = pd.read_csv(csv_path)
    sentimentScores = []
    for tweet in tweets_df['Text']:
        # Getting sentiment score for each of the tweet
        processedTweet = clean_tweets(tweet)
        sentimentScores.append(sentiment_analyzer_scores(processedTweet))
    tweets_df['sentimentScore'] = sentimentScores
    tweets_df.to_csv(csv_path)

# adding_sentiment_score()

def get_avg_sentiment_score_for_a_range_of_days(tweeter_key_word='BitCoin', start_date=datetime.datetime(2015, 1, 1), numdays=18455, top_n_tweets=100, sentiment_score_csv='btc_sentiment_score.csv'):
    '''
        This method is using snscrape to scrape the latest 10000 tweets from Tweeter for giving range dates. 
        It then get the top_n_tweets most liked & retweeted tweets, get a polary sentiment score for each tweets, avg out the sentiment score fot the date.
        Lastly, it saves the sentiment score data for each date into a dataframe.

        Because VADER doens't work with non-English languages. All the rows with lang!=en has being removed from the dataframe. 
        Using translation web api could be a solution for enabling VADER to work on non-english words. But the accuracy will depends on the translation machine and vader

        start_date:             date range start date
        numdays:                number of days from start_date
        top_n_tweets:           top n number of most liked & retweeted tweets
        sentiment_score_csv:    csv file name to save the sentiment scores for each of the date
    '''
    
    sentimentList = []
    # Creating list to append tweet data to
    tweets_list = []
    dateList = [start_date + datetime.timedelta(days=x) for x in range(numdays)]
    numberOfTweetsToRetrieve = 10000

    for date in dateList:
        logging.info('Getting tweets for date {}'.format(date))
        # Using TwitterSearchScraper to scrape data and append tweets to list
        requestString = '{} since:{} until:{}'.format(tweeter_key_word, date.strftime("%Y-%m-%d"), (date + datetime.timedelta(days=1)).strftime("%Y-%m-%d"))
        response = sntwitter.TwitterSearchScraper(requestString).get_items()
        for i,tweet in enumerate(response):
            if i % 1000 == 0 and i / 1000 >=1:
                logging.info('{} tweets got'.format(i))
            if i>numberOfTweetsToRetrieve:
                break
            else:
                tweets_list.append([tweet.date, tweet.id, tweet.content, tweet.user.username, tweet.likeCount, tweet.retweetCount, tweet.lang])

        # Creating a dataframe from the tweets list above
        tweets_df_child = pd.DataFrame(tweets_list, columns=['Datetime', 'Tweet Id', 'Text', 'Username', 'likeCount', 'retweetCount', 'lang'])
        tweets_df_child = tweets_df_child[tweets_df_child.lang == 'en']
        tweets_df_child = tweets_df_child.set_index(['likeCount', 'retweetCount'])
        tweets_df_child = tweets_df_child.sort_index(ascending=False)
        tweets_df_child = tweets_df_child[:top_n_tweets]
        logging.info('df create for date {}. df len is {}'.format(date, len(tweets_df_child)))

        sentimentScore = 0
        tweets = tweets_df_child['Text'].tolist()
        for i in range(top_n_tweets):
            # Get sentiment score for each row in tweets_df_child
            processedTweet = clean_tweets(tweets[i])
            sentimentScore += sentiment_analyzer_scores(processedTweet)
        # Get avg sentiment score for the date
        avgSentimentScore = sentimentScore / top_n_tweets
        # Save the avg score into a dataframe
        sentimentList.append([date, avgSentimentScore])

    sentiment_df = pd.DataFrame(sentimentList,columns=['Datetime', 'sentimentScore'])
    sentiment_df.to_csv('./Data/{}'.format(sentiment_score_csv))

# get_avg_sentiment_score_for_a_range_of_days(numdays=5, top_n_tweets=10, start_date=datetime.datetime(2021, 1, 1))