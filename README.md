# 808_machine-learning-for-business-analytics Note

lec1: Linear Regression

lec2: Logistic Regression

Lec3: Logistic Regression 2

Lec4: Evaluation of model (precision and recall)

Lec5: Keras Learning

Lec6: Tensorflow Learning

Lec7: Unsupervised Learning( K-means and DBSCAN)

Lec8_2: Unsupervised Learning(PCA)

Lec9: Random Forest

Lec9_2: System Recommendation

Lec10: Convolutional Neural Network

project: dogs and cats image recognition

# Other resources

### Over 150 of the Best Machine Learning, NLP, and Python Tutorials I’ve Found
https://medium.com/machine-learning-in-practice/over-150-of-the-best-machine-learning-nlp-and-python-tutorials-ive-found-ffce2939bd78

https://medium.com/machine-learning-in-practice/over-200-of-the-best-machine-learning-nlp-and-python-tutorials-2018-edition-dd8cf53cb7dc

https://medium.com/machine-learning-in-practice/over-200-of-the-best-machine-learning-nlp-and-python-tutorials-2018-edition-dd8cf53cb7dc

### Regularization
what's the difference between L1 and L2?


https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c
http://machinelearningspecialist.com/machine-learning-interview-questions-q8-l1-and-l2-regularization/

### online resourses to learn ML
https://www.springboard.com/blog/machine-learning-online-courses/

###  Quetions for ML
https://www.springboard.com/blog/machine-learning-interview-questions/

### 20 Questions for python 
https://www.springboard.com/blog/python-interview-questions/

### Start up metrics
https://www.slideshare.net/dmc500hats/startup-metrics-for-pirates-long-version


# Machine learning 
### random forest
A random forest classifier.

A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if bootstrap=True (default).

Parameters:
* n_estimators : integer, optional (default=10)

The number of trees in the forest.

* max_depth : integer or None, optional (default=None)
The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

* min_samples_split : int, float, optional (default=2)
The minimum number of samples required to split an internal node

* min_samples_leaf : int, float, optional (default=1)

The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.

* min_weight_fraction_leaf : float, optional (default=0.)

The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.

* max_features : int, float, string or None, optional (default=”auto”)

The number of features to consider when looking for the best split:

* max_leaf_nodes : int or None, optional (default=None)

Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.


* random_state : int, RandomState instance or None, optional (default=None)

If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.


### Cross Valdation
A good way to evaluate a model is to use cross-validation.
K-fold cross-validaton means splitting the training set into K-folds, then making predictions and evaluating them on each fold using a model trained on the remaining folds.

e.g.
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy") array([ 0.9502 , 0.96565, 0.96495])

here we use 3 folds.


What cross-validation technique would you use on a time series dataset?

https://medium.com/@samuel.monnier/cross-validation-tools-for-time-series-ffa1a5a09bf9

we'll want to do something like forward chaining where you’ll be able to model on past data then look at forward-facing data.

* fold 1 : training [1], test [2]
* fold 2 : training [1 2], test [3]
* fold 3 : training [1 2 3], test [4]
* fold 4 : training [1 2 3 4], test [5]
* fold 5 : training [1 2 3 4 5], test [6]


### Handle an imbalanced dataset

An imbalanced dataset is when you have, for example, a classification test and 90% of the data is in one class. That leads to problems: an accuracy of 90% can be skewed if you have no predictive power on the other category of data! Here are a few tactics to get over the hump:

* 1- Collect more data to even the imbalances in the dataset.

* 2- Resample the dataset to correct for imbalances.

* 3- Try a different algorithm altogether on your dataset.

What’s important here is that you have a keen sense for what damage an unbalanced dataset can cause, and how to balance that.

8 Tactics to Combat Imbalanced Classes in Your Machine Learning Dataset
https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/


### Seaborn & Matplotlib

Seaborn and Matplotlib are two of Python's most powerful visualization libraries. Seaborn uses fewer syntax and has stunning default themes and Matplotlib is more easily customizable through accessing the classes.The seaborn package was developed based on the Matplotlib library. It is used to create more attractive and informative statistical graphics. 

### XGboost 

https://towardsdatascience.com/https-medium-com-vishalmorde-xgboost-algorithm-long-she-may-rein-edd9f99be63d

https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/
the code for comparation of XGboost and LightGBM

### LightGBM

https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc

Light GBM is a gradient boosting framework that uses tree based learning algorithm.

* Can we use Light GBM everywhere?
No, it is not advisable to use LGBM on small datasets. Light GBM is sensitive to overfitting and can easily overfit small data. Their is no threshold on the number of rows but my experience suggests me to use it only for data with 10,000+ rows.

* We briefly discussed the concept of Light GBM, now what about it’s implementation?
Implementation of Light GBM is easy, the only complicated thing is parameter tuning. Light GBM covers more than 100 parameters but don’t worry, you don’t need to learn all.
It is very important for an implementer to know atleast some basic parameters of Light GBM. If you carefully go through following parameters of LGBM, I bet you will find this powerful algorithm a piece of cake.

https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc

### Gradient Boosting
https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d

### Brief description for Ensemble, Bagging and Boosting
When we try to predict the target variable using any machine learning technique, the main causes of difference in actual and predicted values are noise, variance, and bias. Ensemble helps to reduce these factors (except noise, which is irreducible error).

An ensemble is just a collection of predictors which come together (e.g. mean of all predictions) to give a final prediction. The reason we use ensembles is that many different predictors trying to predict same target variable will perform a better job than any single predictor alone. Ensembling techniques are further classified into Bagging and Boosting.

* Bagging is a simple ensembling technique in which we build many independent predictors/models/learners and combine them using some model averaging techniques. (e.g. weighted average, majority vote or normal average)

We typically take random sub-sample/bootstrap of data for each model, so that all the models are little different from each other. Each observation is chosen with replacement to be used as input for each of the model. So, each model will have different observations based on the bootstrap process. Because this technique takes many uncorrelated learners to make a final model, it reduces error by reducing variance. Example of bagging ensemble is Random Forest models.

* Boosting is an ensemble technique in which the predictors are not made independently, but sequentially.
This technique employs the logic in which the subsequent predictors learn from the mistakes of the previous predictors. Therefore, the observations have an unequal probability of appearing in subsequent models and ones with the highest error appear most. (So the observations are not chosen based on the bootstrap process, but based on the error). The predictors can be chosen from a range of models like decision trees, regressors, classifiers etc. Because new predictors are learning from mistakes committed by previous predictors, it takes less time/iterations to reach close to actual predictions. But we have to choose the stopping criteria carefully or it could lead to overfitting on training data. Gradient Boosting is an example of boosting algorithm.

More useful resources about Gradient Boosting
My github repo and kaggle kernel link for GBM from scratch:

https://www.kaggle.com/grroverpr/gradient-boosting-simplified/

https://nbviewer.jupyter.org/github/groverpr/Machine-Learning/blob/master/notebooks/01_Gradient_Boosting_Scratch.ipynb

A detailed and intuitive explanation of gradient boosting: How to explain gradient boosting by Terence Parr and Jeremy Howard
Fast.ai github repo link for DecisionTree from scratch (Massive ML/DL related resources): 

https://github.com/fastai/fastai

### Pipelines
Pipelines are a simple way to keep your data processing and modeling code organized. Specifically, a pipeline bundles preprocessing and modeling steps so you can use the whole bundle as if it were a single step.

Many data scientists hack together models without pipelines, but Pipelines have some important benefits. Those include:

Cleaner Code: You won't need to keep track of your training (and validation) data at each step of processing. Accounting for data at each step of processing can get messy. With a pipeline, you don't need to manually keep track of each step.
Fewer Bugs: There are fewer opportunities to mis-apply a step or forget a pre-processing step.
Easier to Productionize: It can be surprisingly hard to transition a model from a prototype to something deployable at scale. We won't go into the many related concerns here, but pipelines can help.
More Options For Model Testing: You will see an example in the next tutorial, which covers cross-validation.

### cross validation

Machine learning is an iterative process.

You will face choices about predictive variables to use, what types of models to use,what arguments to supply those models, etc. We make these choices in a data-driven way by measuring model quality of various alternatives.

You've already learned to use train_test_split to split the data, so you can measure model quality on the test data. Cross-validation extends this approach to model scoring (or "model validation.") Compared to train_test_split, cross-validation gives you a more reliable measure of your model's quality, though it takes longer to run.

### NLP

How to solve 90% of NLP problems: a step-by-step guide
https://blog.insightdatascience.com/how-to-solve-90-of-nlp-problems-a-step-by-step-guide-fda605278e4e

Machine Learning, NLP: Text Classification using scikit-learn, python and NLTK
https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a






