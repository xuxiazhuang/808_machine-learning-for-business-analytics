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

### Regularization
what's the difference between L1 and L2?


https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c
http://machinelearningspecialist.com/machine-learning-interview-questions-q8-l1-and-l2-regularization/

### online resourses to learn ML
https://www.springboard.com/blog/machine-learning-online-courses/


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


# Cross Valdation
A good way to evaluate a model is to use cross-validation.
K-fold cross-validaton means splitting the training set into K-folds, then making predictions and evaluating them on each fold using a model trained on the remaining folds.

e.g.
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy") array([ 0.9502 , 0.96565, 0.96495])
here we use 3 folds.


