#Running ML algorithms and models on the data

import pandas as pd
import numpy as np
import csv
from math import sqrt
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Regression chart.
def chart_regression(pred, y, modelName, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.plot(t['y'].tolist(), label='expected')
    plt.ylabel(modelName)
    plt.legend()
    plt.show()

#Linear regression test
def linearRegression(X_train, y_train, X_test, y_test):
	# Create linear regression object
	regr = LinearRegression()
	# Train the model using the training sets
	regr.fit(X_train, y_train)
	# Make predictions using the testing set
	lin_pred = regr.predict(X_test)
	linear_regression_score = regr.score(X_test, y_test)
	linear_regression_score
	# The coefficients
	print('Coefficients: \n', regr.coef_)
	# The mean squared error
	linRMSE = sqrt(mean_squared_error(y_test, lin_pred))
	print("Root mean squared error: %.2f"
	      % linRMSE)
	# The absolute squared error
	print("Mean absolute error: %.2f"
	      % mean_absolute_error(y_test, lin_pred))
	# Explained variance score: 1 is perfect prediction
	print('R-squared linear regression: %.2f' % r2_score(y_test, lin_pred))
	plt.scatter(y_test, lin_pred)
	plt.xlabel('Measured')
	plt.ylabel('Predicted')
	plt.title('Linear Regression Predicted vs Actual')
	plt.show()
	chart_regression(lin_pred,y_test,'Linear Regression')
	return linear_regression_score, linRMSE


def NeuralNetworkRegression(X_train, y_train, X_test, y_test):
	# Create MLPRegressor object
	mlp = MLPRegressor()
	# Train the model using the training sets
	mlp.fit(X_train, y_train)
	# Score the model
	neural_network_regression_score = mlp.score(X_test, y_test)
	neural_network_regression_score
	# Make predictions using the testing set
	nnr_pred = mlp.predict(X_test)
	# The mean squared error
	nnrRMSE = sqrt(mean_squared_error(y_test, nnr_pred))
	print("Root mean squared error: %.2f"
	      % nnrRMSE)
	# The absolute squared error
	print("Mean absolute error: %.2f"
	      % mean_absolute_error(y_test, nnr_pred))
	# Explained variance score: 1 is perfect prediction
	print('R-squared neural_network: %.2f' % r2_score(y_test, nnr_pred))
	plt.scatter(y_test, nnr_pred)
	plt.xlabel('Measured')
	plt.ylabel('Predicted')
	plt.title('Neural Network Regression Predicted vs Actual')
	plt.show()
	chart_regression(nnr_pred,y_test,'Neural Network Regression')
	return neural_network_regression_score, nnrRMSE


def LassoPrediction(X_train, y_train, X_test, y_test):
	lasso = Lasso()
	lasso.fit(X_train, y_train)
	# Score the model
	lasso_score = lasso.score(X_test, y_test)
	lasso_score
	# Make predictions using the testing set
	lasso_pred = lasso.predict(X_test)
	lassoRMSE= sqrt(mean_squared_error(y_test, lasso_pred))
	print("Root mean squared error: %.2f"% lassoRMSE)
	print('R-squared lasso: %.2f' % r2_score(y_test, lasso_pred))
	plt.scatter(y_test, lasso_pred)
	plt.xlabel('Measured')
	plt.ylabel('Predicted')
	plt.title('Lasso Predicted vs Actual')
	plt.show()
	chart_regression(lasso_pred,y_test,'lasso')
	return lasso_score, lassoRMSE


def ElasticNetPrediction(X_train, y_train, X_test, y_test):
	elasticnet = ElasticNet()
	elasticnet.fit(X_train, y_train)
	elasticnet_score = elasticnet.score(X_test, y_test)
	elasticnet_score
	elasticnet_pred = elasticnet.predict(X_test)
	# The mean squared error
	elasticnetRMSE= sqrt(mean_squared_error(y_test, elasticnet_pred))
	print("Root mean squared error: %.2f"% elasticnetRMSE)
	print('R-squared elasticnet: %.2f' % r2_score(y_test, elasticnet_pred))
	chart_regression(elasticnet_pred,y_test,'ElasticNetPrediction')
	return elasticnet_score, elasticnetRMSE


def DecisionForestRegression(X_train, y_train, X_test, y_test):
	# Create Random Forrest Regressor object
	regr_rf = RandomForestRegressor(n_estimators=200, random_state=1234)
	# Train the model using the training sets
	regr_rf.fit(X_train, y_train)
	# Score the model
	decision_forest_score = regr_rf.score(X_test, y_test)
	decision_forest_score #no print statements? 
	# Make predictions using the testing set
	regr_rf_pred = regr_rf.predict(X_test)
	# The mean squared error
	regrRMSE = sqrt(mean_squared_error(y_test, regr_rf_pred))
	print("Root mean squared error: %.2f"% regrRMSE)
	# The absolute squared error
	print("Mean absolute error: %.2f"% mean_absolute_error(y_test, regr_rf_pred))
	# Explained variance score: 1 is perfect prediction
	print('R-squared decision forest: %.2f' % r2_score(y_test, regr_rf_pred))
	X.columns
	features = X.columns
	importances = regr_rf.feature_importances_
	indices = np.argsort(importances)

	plt.title('Feature Importances')
	plt.barh(range(len(indices)), importances[indices], color='b', align='center')
	plt.yticks(range(len(indices)), features[indices])
	plt.xlabel('Relative Importance')
	plt.show()
	plt.scatter(y_test, regr_rf_pred)
	plt.xlabel('Measured')
	plt.ylabel('Predicted')
	plt.title('Decision Forest Predicted vs Actual')
	plt.show()
	chart_regression(regr_rf_pred,y_test,'Decision Forest Regression')
	return decision_forest_score, regrRMSE


def ExtraTreesPredictor(X_train, y_train, X_test, y_test):
	extra_tree = ExtraTreesRegressor(n_estimators=200, random_state=1234)
	extra_tree.fit(X_train, y_train)
	extratree_score = extra_tree.score(X_test, y_test)
	extratree_score
	extratree_pred = extra_tree.predict(X_test)
	extratreeRMSE = sqrt(mean_squared_error(y_test, extratree_pred))
	print("Root mean squared error: %.2f"% extratreeRMSE)
	print('R-squared extra trees: %.2f' % r2_score(y_test, extratree_pred))

	features = X.columns
	importances = extra_tree.feature_importances_
	indices = np.argsort(importances)

	plt.title('Feature Importances')
	plt.barh(range(len(indices)), importances[indices], color='b', align='center')
	plt.yticks(range(len(indices)), features[indices])
	plt.xlabel('Relative Importance')
	plt.show()
	plt.scatter(y_test, extratree_pred)
	plt.xlabel('Measured')
	plt.ylabel('Predicted')
	plt.title('Extra Trees Predicted vs Actual')
	plt.show()
	chart_regression(extratree_pred,y_test,'ExtraTrees Predictor')
	return extratree_score, extratreeRMSE


def DecisionTreeAdaBoost(X_train, y_train, X_test, y_test):
	# Create Decision Tree Regressor object
	tree_1 = DecisionTreeRegressor()
	tree_2 = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=200, learning_rate=.1)
	# Train the model using the training sets
	tree_1.fit(X_train, y_train)
	tree_2.fit(X_train, y_train)
	# Score the decision tree model
	tree_1.score(X_test, y_test)
	# Score the boosted decision tree model
	boosted_tree_score = tree_2.score(X_test, y_test)
	boosted_tree_score
	# Make predictions using the testing set
	tree_1_pred = tree_1.predict(X_test)
	tree_2_pred = tree_2.predict(X_test)
	# The mean squared error
	tree2RMSE= sqrt(mean_squared_error(y_test, tree_2_pred))
	print("Root mean squared error: %.2f"
	      % tree2RMSE)
	# The absolute squared error
	print("Mean absolute error: %.2f"
	      % mean_absolute_error(y_test, tree_2_pred))
	# Explained variance score: 1 is perfect prediction
	print('R-squared decision tree: %.2f' % r2_score(y_test, tree_2_pred))
	features = X.columns
	importances = tree_2.feature_importances_
	indices = np.argsort(importances)

	plt.title('Feature Importances')
	plt.barh(range(len(indices)), importances[indices], color='b', align='center')
	plt.yticks(range(len(indices)), features[indices])
	plt.xlabel('Relative Importance')
	plt.show()
	plt.scatter(y_test, tree_1_pred)
	plt.xlabel('Measured')
	plt.ylabel('Predicted')
	plt.title('Decision Tree Predicted vs Actual')
	plt.show()
	chart_regression(tree_1_pred,y_test,'Decision tree')

	plt.scatter(y_test, tree_2_pred)
	plt.xlabel('Measured')
	plt.ylabel('Predicted')
	plt.title('Boosted Decision Tree Predicted vs Actual')
	plt.show()
	chart_regression(tree_2_pred,y_test,'Adaboost + DT')

	return boosted_tree_score, tree2RMSE

def XGBoostPredictor(X_train, y_train, X_test, y_test):
	#Fitting XGB regressor 
	xboost = XGBRegressor(n_estimators=200)
	xboost.fit(X_train, y_train)
	xgb_score = xboost.score(X_test, y_test)
	xgb_score
	#Predict 
	xboost_pred = xboost.predict(X_test)
	xgboostRMSE= sqrt(mean_squared_error(y_test, xboost_pred))
	print("Root mean squared error: %.2f"% xgboostRMSE)
	print('R-squared fir XGBoost : %.2f' % r2_score(y_test, xboost_pred))
	plt.scatter(y_test, xboost_pred)
	plt.xlabel('Measured')
	plt.ylabel('Predicted')
	plt.title('XGBoost Predicted vs Actual')
	plt.show()
	chart_regression(xboost_pred,y_test,'XGBoost Predictor')
	return xgb_score, xgboostRMSE


def knnModel(X_train, y_train, X_test, y_test):
	knn = KNeighborsRegressor(n_neighbors=5)
	# Fit the model on the training data.
	knn.fit(X_train, y_train)
	# Make point predictions on the test set using the fit model.
	knn_score = knn.score(X_test, y_test)
	knnpred = knn.predict(X_test)
	# mse = (((predictions - actual) ** 2).sum()) / len(predictions)
	knnRMSE = sqrt(mean_squared_error(y_test, knnpred))
	print("Root mean squared error: %.2f"% knnRMSE)
	print('R-squared for knn: %.2f' % r2_score(y_test, knnpred))
	plt.scatter(y_test, knnpred)
	plt.xlabel('Measured')
	plt.ylabel('Predicted')
	plt.title('knn Predicted vs Actual')
	plt.show()
	chart_regression(knnpred,y_test,'knn Model')
	return knn_score, knnRMSE

def RidgeModel(X_train, y_train, X_test, y_test):
	rid = Ridge(alpha=1.0)
	rid.fit(X_train, y_train)
	ridge_score=rid.score(X_test,y_test)
	ridgePred = rid.predict(X_test)
	ridgeRMSE = sqrt(mean_squared_error(y_test, ridgePred))
	print("Root mean squared error: %.2f"% ridgeRMSE)
	print('R-squared for ridge model: %.2f' % r2_score(y_test, ridgePred))
	plt.scatter(y_test, ridgePred)
	plt.xlabel('Measured')
	plt.ylabel('Predicted')
	plt.title('Ridge Predicted vs Actual')
	plt.show()
	chart_regression(ridgePred,y_test,'Ridge Model')
	return ridge_score, ridgeRMSE


# def SVMmodel(X_train, y_train, X_test, y_test):
# 	svm1 = SVR(gamma='scale', C=1.0, epsilon=0.2)
# 	svm1.fit(X_train, y_train) 
# 	svm_score=svm1.score(X_test,y_test)
# 	svmPred = svm1.predict(X_test)
# 	svmRMSE = sqrt(mean_squared_error(y_test, svmPred))
# 	print("Root mean squared error: %.2f"% svmRMSE)
# 	plt.scatter(y_test, svmPred)
# 	plt.xlabel('Measured')
# 	plt.ylabel('Predicted')
# 	plt.title('SVM Predicted vs Actual')
# 	plt.show()
# 	return svm_score, svmRMSE


if __name__=="__main__":
	df = pd.read_csv('weatherAndAQIdelhi_cleaned.csv') 
	y=df['AQI']
	df = df.drop(df.columns[[0]], axis=1)  # df.columns is zero-based pd.Index 
	X=df.drop(['AQI', 'date'],axis=1)  #, 'conds', 'conds_1','conds_2','conds_3','conds_4','conds_5'

	#WE NEED TO ADD THE CONDNS FACTOR ALSO AS WELL

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=1234)

	print(X_train.shape) 
	print(y_train.shape)

	print(X_test.shape) 
	print(y_test.shape)

	linear_regression_score, linRMSE=linearRegression(X_train, y_train, X_test, y_test)
	neural_network_regression_score, nnrRMSE =NeuralNetworkRegression(X_train, y_train, X_test, y_test)
	lasso_score, lassoRMSE=LassoPrediction(X_train, y_train, X_test, y_test)
	elasticnet_score, elasticnetRMSE=ElasticNetPrediction(X_train, y_train, X_test, y_test)
	decision_forest_score, regrRMSE=DecisionForestRegression(X_train, y_train, X_test, y_test)
	extratree_score, extratreeRMSE=ExtraTreesPredictor(X_train, y_train, X_test, y_test)
	boosted_tree_score, tree2RMSE =DecisionTreeAdaBoost(X_train, y_train, X_test, y_test)
	xgb_score, xgboostRMSE =XGBoostPredictor(X_train, y_train, X_test, y_test)
	knn_score, knnRMSE = knnModel(X_train, y_train, X_test, y_test)
	ridge_score, ridgeRMSE = RidgeModel(X_train, y_train, X_test, y_test)
	# svm_score, svmRMSE = SVMmodel(X_train, y_train, X_test, y_test)

	print("Scores:")
	print("Linear regression score: ", linear_regression_score)
	print("Neural network regression score: ", neural_network_regression_score)
	print("Lasso regression score: ", lasso_score)
	print("ElasticNet regression score: ", elasticnet_score)
	print("Decision forest score: ", decision_forest_score)
	print("Extra Trees score: ", extratree_score)
	print("Boosted decision tree score: ", boosted_tree_score)
	print("XGBoost score:", xgb_score)
	print("knn score:", knn_score)
	print("Ridge score:", ridge_score)
	# print("SVM score:", svm_score)

	print("\n")
	print("RMSE:")
	print("Linear regression RMSE: %.2f"% linRMSE)
	print("Neural network RMSE: %.2f"% nnrRMSE)
	print("Lasso RMSE: %.2f"% lassoRMSE)
	print("ElasticNet RMSE: %.2f"% elasticnetRMSE)
	print("Decision forest RMSE: %.2f"% regrRMSE)
	print("Extra Trees RMSE: %.2f"% extratreeRMSE)
	print("Boosted decision tree RMSE: %.2f"% tree2RMSE)
	print("XGBoost RMSE: %.2f"% xgboostRMSE)
	print("knn RMSE: %.2f"% knnRMSE)
	print("Ridge RMSE: %.2f"% ridgeRMSE)
	# print("SVM RMSE: %.2f"% svmRMSE)
	