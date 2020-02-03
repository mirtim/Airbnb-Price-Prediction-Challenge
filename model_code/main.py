#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from scipy.spatial import distance

df = pd.read_csv('data/data.csv')
trainData = pd.read_csv('data/train.csv')
testData = pd.read_csv('data/test.csv')
valData = pd.read_csv('data/val.csv')


# first data cleaning and model

testData1 = testData.copy()

cleanedData1 = df[['id', 'latitude', 'longitude', 'host_listings_count', 'accommodates', 'number_of_reviews', 'beds', 'review_scores_rating', 'bathrooms', 'number_of_reviews_ltm', 'bedrooms', 'price']]

cleanedData1.loc[:, ['latitude']] = cleanedData1[['latitude']] / max(cleanedData1['latitude'])
cleanedData1.loc[:, ['longitude']] = cleanedData1[['longitude']] / min(cleanedData1['longitude'])
cleanedData1.loc[:, ['host_listings_count']] = cleanedData1[['host_listings_count']] / max(cleanedData1['host_listings_count'])
cleanedData1.loc[:, ['accommodates']] = cleanedData1[['accommodates']] / max(cleanedData1['accommodates'])
cleanedData1.loc[:, ['number_of_reviews']] = cleanedData1[['number_of_reviews']] / max(cleanedData1['number_of_reviews'])
cleanedData1.loc[:, ['beds']] = cleanedData1[['beds']] / max(cleanedData1['beds'])
cleanedData1.loc[:, ['review_scores_rating']] = cleanedData1[['review_scores_rating']] / max(cleanedData1['review_scores_rating'])
cleanedData1.loc[:, ['bathrooms']] = cleanedData1[['bathrooms']] / max(cleanedData1['bathrooms'])
cleanedData1.loc[:, ['number_of_reviews_ltm']] = cleanedData1[['number_of_reviews_ltm']] / max(cleanedData1['number_of_reviews_ltm'])
cleanedData1.loc[:, ['bedrooms']] = cleanedData1[['bedrooms']] / max(cleanedData1['bedrooms'])

cleanedTrain1 = cleanedData1.merge(trainData, on=["id"])
cleanedTrain1.fillna(cleanedData1.mean())
cleanedTrain1 = cleanedTrain1.sample(frac=1,random_state=0)

cleanedTest1 = cleanedData1.merge(testData, on=["id"])
cleanedTest1 = cleanedTest1.drop("price", axis=1)
cleanedTest1.fillna(cleanedData1.mean())

cleanedVal1 = cleanedData1.merge(valData, on=["id"])
cleanedVal1.fillna(cleanedData1.mean())




def predictPriceMultivariate1(newListingValue,featureColumns):
    tempDf1 = cleanedTrain1
    tempDf1['distance'] = distance.cdist(tempDf1[featureColumns],[newListingValue[featureColumns]])
    tempDf1 = tempDf1.sort_values('distance')
    knn19 = tempDf1.price.iloc[:19]
    predictedPrice1 = knn19.mean()
    return predictedPrice1



cols = ['latitude', 'longitude', 'host_listings_count', 'accommodates', 'number_of_reviews']
cleanedTest1['predictedPrice'] = cleanedTest1[cols].apply(predictPriceMultivariate1,featureColumns=cols,axis=1)

testData1['price'] = cleanedTest1['predictedPrice']
# testData.to_csv(r'C:\Users\timpc\Desktop\CS542\Challenge\predictions2.csv', index=False)


# second data cleaning and model

testData2 = testData.copy()

cleanedData2 = df[['id', 'latitude', 'longitude', 'host_listings_count', 'accommodates', 'number_of_reviews', 'beds', 'review_scores_rating', 'bathrooms', 'number_of_reviews_ltm', 'bedrooms', 'price']]
cleanedData2.apply(pd.to_numeric)

cleanedData2.loc[:, ['latitude']] = (cleanedData2[['latitude']] - cleanedData2[['latitude']].min()) / (cleanedData2[['latitude']].max() - cleanedData2[['latitude']].min())
cleanedData2.loc[:, ['longitude']] = (cleanedData2[['longitude']] - cleanedData2[['longitude']].min()) / (cleanedData2[['longitude']].max() - cleanedData2[['longitude']].min())
cleanedData2.loc[:, ['host_listings_count']] = (cleanedData2[['host_listings_count']] - cleanedData2[['host_listings_count']].min()) / (cleanedData2[['host_listings_count']].max() - cleanedData2[['host_listings_count']].min())
cleanedData2.loc[:, ['accommodates']] = (cleanedData2[['accommodates']] - cleanedData2[['accommodates']].min()) / (cleanedData2[['accommodates']].max() - cleanedData2[['accommodates']].min())
cleanedData2.loc[:, ['number_of_reviews']] = (cleanedData2[['number_of_reviews']] - cleanedData2[['number_of_reviews']].min()) / (cleanedData2[['number_of_reviews']].max() - cleanedData2[['number_of_reviews']].min())
cleanedData2.loc[:, ['beds']] = (cleanedData2[['beds']] - cleanedData2[['beds']].min()) / (cleanedData2[['beds']].max() - cleanedData2[['beds']].min())
cleanedData2.loc[:, ['review_scores_rating']] = (cleanedData2[['review_scores_rating']] - cleanedData2[['review_scores_rating']].min()) / (cleanedData2[['review_scores_rating']].max() - cleanedData2[['review_scores_rating']].min())
cleanedData2.loc[:, ['bathrooms']] = (cleanedData2[['bathrooms']] - cleanedData2[['bathrooms']].min()) / (cleanedData2[['bathrooms']].max() - cleanedData2[['bathrooms']].min())
cleanedData2.loc[:, ['number_of_reviews_ltm']] = (cleanedData2[['number_of_reviews_ltm']] - cleanedData2[['number_of_reviews_ltm']].min()) / (cleanedData2[['number_of_reviews_ltm']].max() - cleanedData2[['number_of_reviews_ltm']].min())
cleanedData2.loc[:, ['bedrooms']] = (cleanedData2[['bedrooms']] - cleanedData2[['bedrooms']].min()) / (cleanedData2[['bedrooms']].max() - cleanedData2[['bedrooms']].min())

cleanedTrain2 = cleanedData2.merge(trainData, on=["id"])
cleanedTrain2.fillna(cleanedData2.mean())
cleanedTrain2 = cleanedTrain2.sample(frac=1,random_state=0)

cleanedTest2 = cleanedData2.merge(testData, on=["id"])
cleanedTest2 = cleanedTest2.drop("price", axis=1)
cleanedTest2.fillna(cleanedData2.mean())

cleanedVal2 = cleanedData2.merge(valData, on=["id"])
cleanedVal2.fillna(cleanedData2.mean())



def predictPriceMultivariate2(newListingValue,featureColumns):
    tempDf2 = cleanedTrain2
    tempDf2['distance'] = distance.cdist(tempDf2[featureColumns],[newListingValue[featureColumns]])
    tempDf2 = tempDf2.sort_values('distance')
    knn18 = tempDf2.price.iloc[:18]
    predictedPrice2 = knn18.mean()
    return predictedPrice2



cols = ['latitude', 'longitude', 'host_listings_count', 'accommodates', 'number_of_reviews']
cleanedTest2['predictedPrice'] = cleanedTest2[cols].apply(predictPriceMultivariate2,featureColumns=cols,axis=1)

testData2['price'] = cleanedTest2['predictedPrice']
# testData.to_csv(r'C:\Users\timpc\Desktop\CS542\Challenge\predictions2.csv', index=False)
testData2.head()


#combining the result into one

df_concat = pd.concat((testData1, testData2))

by_row_index = df_concat.groupby(df_concat.index)
pred = by_row_index.mean()


pred.to_csv('pred.csv', index=False)
