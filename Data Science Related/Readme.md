## [Data-Driven Fantasy Premier League(FPL)](https://github.com/sob-ANN/Projects/blob/main/Data%20Science%20Related/EDA%20FPL.ipynb)
Fantasy Premier League (FPL) is played by over 10 Million football fans across the globe each year. It can be highly competitive, as I found out after years of playing and struggling. Here, I attempt to try and make use of data from the FPL website using an API call, to make data-driven decisions in selecting my team. Side note: This was done in the off season when the previous season had already finished and I had a year's data on all the players. I created certain metrics like 'Value' = points scored/cost, which indicates how cost effective a player is. I tried to find the players with most value in each position. Another metric I used was xG and xA which stands for Expected Goals and Expected Assists. xG measures the probability of a shot resulting in a goal based on previous shots taken from the same position and a bunch of other things. It is said that xG will catch up to you eventually. This is where we can 'look beyond the underlying numbers'. Here, I was looking for players who outperfomed their xG (i.e. Goals - xG > 0). Although it is good that they perform better than what is expected of them, they may not keep performing on the same level next season. Similarly, players that underperfomed may be expected to perform better in the next season. Based on this, we can make more informed decisions. 
Finally, I attempted to find the 'most optimal' team based on the various constraints in the game like budget, max/min player per position and max per team. I would like to continue this project in the future to include in-season changes and maybe predictions. Also, previous year data may be helpful to use. This was my naive attempt to put to practice what I have learnt.

## [Sentiment Analysis RNN, LSTM](https://github.com/sob-ANN/Projects/blob/main/Data%20Science%20Related/Sentiment_Analysis_RNN_LSTM.ipynb)
___WRITE___

## [Neural Network Tensorflow](https://github.com/sob-ANN/Projects/blob/main/Data%20Science%20Related/Neural%20Network%20Tensorflow.ipynb)
A basic implementation of Neural Network in Tensorflow on MNIST dataset. Here we were trying to see what representation is learnt by the hidden layers while learning how to apply the TensorFlow API.

## [Support Vector Machine](https://github.com/sob-ANN/Projects/blob/main/Data%20Science%20Related/Support%20Vector%20Machine.ipynb)
Support Vector Machine, a classification Algorithm was implemented using sklearn library. Here, we made use of various kernel functions - rbf, polynomial etc. A lot of hyperparameter tuning was done to improve accuracy. Finally a Kaggle competetion was hosted (https://www.kaggle.com/competitions/ell-784-assignment-2/leaderboard) where my model had an accuracy of 0.98 (2022AMY7554)

## [K-means Clustering](https://github.com/sob-ANN/Projects/blob/main/Data%20Science%20Related/K%20means%20Clustering.ipynb)
K-means Clustering is an unsupervised clustering algorithm. This model was implemented using sklearn on MNIST datset. We experimented with the value of k (number of clusters) to see how it affects our predictions.

