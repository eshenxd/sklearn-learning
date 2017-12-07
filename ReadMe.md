### 学习目的
sklearn是一个Python机器学习包，其中包含了大部分的机器学习算法，由于最近小组的项目越来越偏向于数据导向，所以打算系统学习一下sklearn包。

### 框架
在网上找到了一个sklearn的脑图：
![image](http://orxe6lzm4.bkt.clouddn.com/YouDao/1512385626126.png) 
目前的学习计划是按照脑图，逐个算法，至少需要搞清楚一下几点内容：
- 算法的适用范围
- 算法的基本原理
- 算法的应用案例
- 算法的训练方法  

sklearn 算法按照目的分，可以分成类别预测和量化

#### 类别预测
类别预测算法根据是否具有标定好的数据，分成分类算法和聚类算法  
1. 分类算法  
分类算法根据数据规模，可以分为linear svc和SGD classifier，二者的分界线是是否具有大于100K的标定数据
    - linear SVC  
    对于文本数据，linear svc方法是Naive Bayes， 对于非文本数据可以使用K近邻分类，如果效果不好，在尝试SVC和Ensemble Classifiers
        + Unsupervised Nearest Neighbors
        ```python
        # code
        from sklearn.neighbors import NearestNeighbors
        import numpy as np
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        # algorithm : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)
        ```
        + Nearest Neighbors Classification
        ```python
        # code
        # weights = {'uniform', 'distance'}
        clf = neighbors.KNeighorsClassifier(n_neighbors, weighs='uniform')
        clf.fit(X, y)
        ```
        + Nearest Neighbors regression
        ```python
        # code
        # weights = {'uniform', 'distance'}
        clf = neighbors.KNeighborsRegressor(n_neighbors, weights = 'distance')
        clf.fit(X, y)
        ```
    - SVC
        + normal
        ```python
        # code
        from sklearn import svm
        clf = svm.SVC()
        clf.fit(X, y)
        ```

        + unbalanced samples problems
        ```python
        # code
        sample_weights = 
        clf_weights = svm.SVC()
        clf_weights.fit(X, y, sample_weight=sample_weights)
        ```
    
    - Ensemble Classifiers
        + Bagging algorithm
        bagging的方法是将数据和特征分别放在黑盒中，随机抽取一些数据和特征，训练一个base estimator的方法
        ```python
        # code
        from sklearn.ensemble import BaggingClassifier
        from sklearn.neignbors import KNeighborsClassifier
        bagging = BaggingClassifier(n_estimators=50, bootstrap=true, KNerighborsClassifier(), bootstrap_feature=True, max_samples=0.5, max_features=0.5)
        bagging.fit(X, y)
        ```
        + RandomForest algorithm
        ```python
        # code classifier
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(max_depth=2, random_state=0)
        clf.fit(X, y)
        
        # code regressor
        from sklearn.ensemble import RandomForestRegressor
        regr = RandomForestRegressor(max_depth=2, random_state=0)
        regr.fit(X, y)
        ```
        + Extremely Randomized Trees
        原先决策树针对是连续数值的特征会计算局部split value，（一个特征可能可以产生多个split value，都计算一下，然后评估所有的特征中的哪一个split value最好，就以该特征的该split value分裂）；但是现在，对每一个特征，在它的特征取值范围内，随机生成一个split value，再计算看选取哪一个特征来进行分裂（树多一层）
        ```python
        # code
        from sklearn.ensemble import ExtraTreesClassifier
        clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
        ```


#### GridSearch
在sk库中，有很多分类和回归方法，这些方法很多时候对同一个问题都是适用的只是效果不同，而这些效果会根据模型超参数的设定而有差别，因此，对于实际问题，有的时候需要尝试多个模型的多个超参数，sk库提供了一个校验方法，GridSearch，代码如下：
```python
from __future__ import print_function

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

print(__doc__)

# Loading the Digits dataset
digits = datasets.load_digits()

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='%s_weighted' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
```

在使用GridSearch时，有几个小的tips：  
- 指定一个客观的指标，用来对比几个算法的性能，比如上面代码中的precision, recall  
- 指定多个评估指标，而不能仅仅对某一个指标  
- 要综合考虑模型和参数的影响  
- 模型选择，划分数据集，分别用来训练和评估模型  
- 并行执行  
- 对失败保持鲁棒性  










### Learning List  
- [ ] pandas  
- [x] GridSearchCV sklearn中提供的用于各类模型调参的方法  
- [ ] sklearn.pipeline  
- [ ] function signature  