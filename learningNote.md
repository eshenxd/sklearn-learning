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










### Learning List  
- [ ] pandas
- [ ] GridSearchCV sklearn中提供的用于各类模型调参的方法