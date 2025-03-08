# Quant project

Here is a quant project I intend to develop, building my own framework from analysis to trading. I will start by using Binance as my broker and data collector

## Advancements :

- I have built the fundation of the trading and data collection infrastructure, but I'm currently looking for an idea to build. I don't have concrete examples of what is possible and how to implement it but I'm looking for it.
    
- The first model is a Chain Classifier on Random Forest to try to estimate the quantile in wich future returns will end up. The accuracy is around 30% on each quantile in a rolling window cross-validation, which is above random but clearly not enough to find an edge.

- The second model is a Random Forest with a Logistic Regression at the end that would take the RF model probability and put it into context with the ATR to see if a certain treshold in the log return would be achieved. This model had poor metrics, and I don't think the logistic regression added improved that much the model. It would make it more prone to trade but the gains don't outweighs the risk raken I believe.

- I also added a permutation test that I would do on in sample and out of sample data. If a model is strong on in sample data, instead of using out of sample directly to confirm it, I will test it on data that is a random permutation of the existing in-sample but where the statistics are the same. The different moments, and even correlation of different dataset permutted are the same, but the time relationship and everything else is destroyed. I will test the model on 100s of permutations, and if the log returns of the strategy on the real data isn't in the top 1%, that means that I got this return by luck or noise, but not by discovering a relationship in the price. The idea would be to test and optimize on real data. Then test on permutations of the in-sample. Then test on out-sample and last test on permutation of the out of sample. This would help me save a lot of data while trying to not biased in my model selection.
