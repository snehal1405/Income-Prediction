# Income-Prediction
Executive Summary

In this case study, the aim is to build a predictive model to determine income levels (below or above 50k) for people in the US. We are dealing with a binary classification problem for predicting a specific outcome that can only take two distinct values for example 0 and 1 or positive and negative etc using imbalance data. 

We obtained the data UCI Machine learning repository here which consists of total 299,285 rows and 41 columns. 
The key independent variables of the dataset are -
Age, Marital Status, Income, Family Members, No. of Dependents, Tax Paid, Investment (Mutual Fund, Stock), Return from Investments, Education, Spouse, Education, Nationality, Occupation, Region in US, Race, Occupation category.
The dependent variable is Income and we have to predict the income range, whether 
the income will be less or greater than 50,000 dollars. 

We used 3 sampling methods to overcome the imbalance problem:
1.	Undersampling- by adding more of the minority class so it has more effect on the machine learning algorithm.
2.	Oversampling-by removing some of the majority class so it has less effect on the machine learning algorithm.
3.	SMOTE-checks n nearest neighbors, measures the distance between them and introduces a new observation at the center of n observations.
Then we apply the following machine learning algorithms on all 3 datasets obtained from different sampling techniques to study the effect and solve binary classification problem:
1.	Naive Bayes Classifier
2.	XGboost
 
In terms of sampling the imbalanced data, SMOTE dominates undersampling and oversampling because of loss of information and overestimation of minority class respectively. For binary classification based on “income_level”, Naive Bayes results in 79% accuracy whereas XGBoost results in 94.8% accuracy. 

