import warnings
from collections import Counter
from itertools import cycle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.cluster import KMeans, AffinityPropagation, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, confusion_matrix, plot_roc_curve, plot_confusion_matrix, PrecisionRecallDisplay, classification_report, make_scorer, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics.cluster import contingency_matrix
from pyclustering.cluster.clarans import clarans
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

def auto_ml(df):
    # Data preprocessing
    df.apply(pd.unique)

    # Classification
    #sns.heatmap(data.corr(), annot=True, fmt='0.2f')
    #plt.show()

    # preprocessed_dataset_list: Index = 0 - Standard, 1 - MinMax, 2 - MaxAbs, 3 - Robust scaled data
    preprocessed_dataset_list, preprocessed_target = encode_scale(df,
                                                                  ["NO_OF_CHILD", "INCOME", "WORK_PHONE", "PHONE",
                                                                   "E_MAIL", "FAMILY SIZE", "BEGIN_MONTH", "AGE",
                                                                   "YEARS_EMPLOYED"],
                                                                  ["GENDER", "CAR", "REALITY", "INCOME_TYPE",
                                                                   "EDUCATION_TYPE", "FAMILY_TYPE", "HOUSE_TYPE",
                                                                   "TARGET"],
                                                                  "TARGET",
                                                                  scalers=[StandardScaler(), MinMaxScaler(),
                                                                           MaxAbsScaler(), RobustScaler()],
                                                                  encoders=[LabelEncoder()])

    # Use SMOTE - handle imbalanced dataset
    # preprocessed_target_smote =
    sm = SMOTE(sampling_strategy='minority')
    preprocessed_dataset_list_smote = []    # smoted data with Index = 0 - Standard, 1 - MinMax, 2 - MaxAbs, 3 - Robust scale
    preprocessed_target_smote = []      # target value of each smoted data
    for x in preprocessed_dataset_list:
        x_smote, y_smote = sm.fit_resample(x, preprocessed_target)
        preprocessed_dataset_list_smote.append(x_smote)
        preprocessed_target_smote.append(y_smote)
    print(Counter(preprocessed_target_smote[0]))

    # Feature selection - Intuition
    intuitive_fs = []
    for x in preprocessed_dataset_list_smote:
        intuitive_fs.append(x[["CAR", "REALITY", "EDUCATION_TYPE", "INCOME", "WORK_PHONE", "PHONE", "AGE"]])

    # Feature selection - SelectKBest
    kBest_fs = []
    print('SelectKBest')
    i = 0
    for x in preprocessed_dataset_list_smote:
        train_X, test_X, train_y, test_y = train_test_split(x, preprocessed_target_smote[i], test_size=0.2)
        model = SelectKBest(score_func=f_classif, k=4)
        kBest = model.fit(train_X, train_y)
        support = kBest.get_support()
        features = x.columns
        print(features[support])
        new_X = x.loc[:, features[support]]
        kBest_fs.append(new_X)
        i += 1

    # Feature selection - RFE
    rfe_fs = []
    print('RFE')
    i = 0
    for x in preprocessed_dataset_list_smote:
        train_X, test_X, train_y, test_y = train_test_split(x, preprocessed_target_smote[i], test_size=0.2)
        model = LogisticRegression()
        rfe = RFE(model, n_features_to_select=4)
        fit = rfe.fit(train_X, train_y)
        support = fit.get_support()
        features = x.columns
        print(features[support])
        new_X = x.loc[:, features[support]]
        rfe_fs.append(new_X)
        i += 1

    # Feature selection - SelectFromModel
    sfm_fs = []
    print('SelectFromModel')
    i = 0
    for x in preprocessed_dataset_list_smote:
        train_X, test_X, train_y, test_y = train_test_split(x, preprocessed_target_smote[i], test_size=0.2)
        selector = SelectFromModel(estimator=LogisticRegression()).fit(train_X, train_y)
        features = x.columns
        print(features[selector.get_support()])
        new_X = x.loc[:, features[selector.get_support()]]
        sfm_fs.append(new_X)
        i += 1

    # Classification
    find_best_classification(kBest_fs, preprocessed_target_smote)

    # Clustering
    intuitive_cluster = findBestCluster(intuitive_fs, preprocessed_target_smote)
    k_best_cluster = findBestCluster(kBest_fs, preprocessed_target_smote)
    rfe_cluster = findBestCluster(rfe_fs, preprocessed_target_smote)
    sfm_cluster = findBestCluster(sfm_fs, preprocessed_target_smote)

    plotCluster(intuitive_fs, intuitive_cluster)
    plotCluster(kBest_fs, k_best_cluster)
    plotCluster(rfe_fs, rfe_cluster)
    plotCluster(sfm_fs, sfm_cluster)


def encode_scale(df, numerical_feature_list, categorical_feature_list, target_name, scalers=None, encoders=None, fill_nan=None, outliers=None):
    """ @params
    predictor_names : List of names of predictor features.
    Ex ) predictor_names  = df.columns[:-1]

    target_name: name of the target column
    Ex) target_name = df.columns[-1]

    fill_nan : Dictionary, The key is "replace" ans "fill_nan". Default = None.
    Ex) fill_nan = {"replace":{"0":np.nan},"fill_nan" : {"Weight":"mean"}

    encoders: Dictionary, key is encoded feature name, value is encoders. Default = None.
    Ex) encoders = { “f_name1” : OneHotEncoder(), “f_name2”: LableEncoder() }

    scalers: A list of various scaler. Default = None.
    Ex) scalers = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]

    outliers: Dictionary, the key is categorical or numerical, the value is upper, lower bound.
    Ex) outliers = { “categorical” : {“f_name1”:”Z”}, “numerical” : {“f_name2” : [100,0] } }
    """

    # Handling missing values
    if fill_nan is not None:
        if "replace" in fill_nan:
            df.replace(fill_nan["replace"], inplace=True)
        if "fill_nan" in fill_nan:
            for key, value in fill_nan["fill_nan"].items():
                if value == "mean":
                    df[key] = df[key].apply(pd.to_numeric)
                    df[key].fillna(df[key].mean(), inplace=True)
                elif value == "max":
                    df[key] = df[key].apply(pd.to_numeric)
                    df[key].fillna(df[key].max(), inplace=True)
                elif value == "min":
                    df[key] = df[key].apply(pd.to_numeric)
                    df[key].fillna(df[key].min(), inplace=True)
                else:
                    df[key].fillna(value, inplace=True)

    # Handling outliers
    if outliers is not None:
        if "categorical" in outliers:
            for key, value in outliers["categorical"].items():
                for outlier in value:
                    df.drop(df[df[key] == outlier].index, inplace=True)

        elif "numerical" in outliers:
            for key, value in outliers["numerical"].items():
                df.drop(df[df[key] > value[0]].index, inplace=True)
                df.drop(df[df[key] < value[1]].index, inplace=True)

    # Encoding and Scaling
    encoded_scaled_df_list = []
    encoded_target = pd.Series(dtype=int)
    if encoders is not None:
        for encoder in encoders:
            df_copy = df.copy()
            if type(encoder) == LabelEncoder:
                df_copy = df_copy.apply(encoder.fit_transform)
                encoded_target = df_copy[target_name]
                df_copy.drop(target_name, axis=1, inplace=True)
                if scalers is not None:
                    for scaler in scalers:
                        df_copy[numerical_feature_list] = scaler.fit_transform(df_copy[numerical_feature_list])
                        encoded_scaled_df_list.append(df_copy)
            else:
                df_copy[categorical_feature_list] = encoder.fit_transform(df_copy[categorical_feature_list])
                encoded_target = df_copy[target_name]
                df_copy.drop(target_name, axis=1, inplace=True)
                if scalers is not None:
                    for scaler in scalers:
                        df_copy[numerical_feature_list] = scaler.fit_transform(df_copy[numerical_feature_list])
                        encoded_scaled_df_list.append(df_copy)

    return encoded_scaled_df_list, encoded_target


def viz_classification(model, X, y, normalize, mode = None):
    """
    :param model:
    :param X:
    :param y:
    :param normalize:
    :param mode:ROC curve, PRC, CAP curve and the confusion matrix.
    :return:
    """
    """ mode - confusion_matrix """
    if "cfm" in mode:
        plot = plot_confusion_matrix(model,X,y,
                                 normalize=normalize)
        plot.ax_.set_title("Confusion Matrix")

    if "roc" in mode:
        plot = plot_roc_curve(model, X, y,)
        plot.ax_.set_title("ROC Curve")

    if "prc" in mode:
        plot = PrecisionRecallDisplay.from_estimator(model,X,y)
        plot.ax_.set_title("Precision-Recall curve")


def find_best_classification(x_list, y):
    """
    find the set of best parameters among classification algorithms according to silhouette score and purity.

    :param x_list: list of data encoded and scaled with specified features
    :param y: target feature

    return
    list[{decisionTree}, {logisticRegression}, {svm}]
    each dict contains best precision, recall, f1 score, params, idx.
    """
    # Models
    models = [DecisionTreeClassifier(), LogisticRegression(), svm.SVC()]
    model_names = ['DecisionTreeClassifier', 'LogisticRegression', 'SVM']

    # Parameter grids
    decision_tree_param_grid = {'criterion': ['entropy', 'gini'], 'max_depth': [3, 4, 5],
                                'min_samples_split': [2, 4, 6], 'min_samples_leaf': [1, 3, 5]}
    logistic_regression_param_grid = {'penalty': ['l2'], 'C': np.logspace(-4, 4, 20),
                                      'solver': ['newton-cg', 'lbfgs', 'liblinear']}
    svm_param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                      'gamma': [0.01, 0.1, 1, 10], 'max_iter': [100, 1000, 10000, -1]}
    param_grids = [decision_tree_param_grid, logistic_regression_param_grid, svm_param_grid]

    # Scoring criteria
    scorers = {'precision_score': make_scorer(precision_score),
               'recall_score': make_scorer(recall_score),
               'f1_score': make_scorer(f1_score)}
    best_score_dict_list = []

    for idx, data_x in enumerate(x_list):
        x_train, x_test, y_train, y_test = train_test_split(data_x, y[idx], shuffle=True, stratify=y)
        print('No.' + str(idx + 1) + ' Dataset')
        for classification_model, model_name, param_grid in zip(models, model_names, param_grids):
            grid_search_cv = GridSearchCV(classification_model, param_grid, scoring=scorers, refit='recall_score', cv=3,
                                          verbose=1)
            grid_search_cv.fit(x_train, y_train)
            y_pred = grid_search_cv.predict(x_test)

            score = pd.DataFrame(grid_search_cv.cv_results_)
            score = score[['params', 'mean_test_precision_score', 'rank_test_precision_score',
                           'mean_test_recall_score', 'rank_test_recall_score', 'mean_test_f1_score',
                           'rank_test_f1_score']]

            best_score_dict = {
                'precision score': score.loc[score.rank_test_precision_score == 1, ['mean_test_precision_score']].iat[
                    0, 0], 'precision params': score.loc[score.rank_test_precision_score == 1, ['params']].values,
                'recall score': score.loc[score.rank_test_recall_score == 1, ['mean_test_recall_score']].iat[0, 0],
                'recall params': score.loc[score.rank_test_recall_score == 1, ['params']].values,
                'F1 score': score.loc[score.rank_test_f1_score == 1, ['mean_test_f1_score']].iat[0, 0],
                'F1 params': score.loc[score.rank_test_f1_score == 1, ['params']].values}

            print('******************Train data results********************')
            print(model_name + '\'s best precision score:\n', best_score_dict['precision score'])
            print(model_name + '\'s best recall score:\n', best_score_dict['recall score'])
            print(model_name + '\'s best f1 score:\n', best_score_dict['F1 score'])
            print()

            print('******************Test data results********************')
            print(classification_report(y_test, y_pred, zero_division=0))
            print(model_name + '\'s best accuracy score:\n', accuracy_score(y_test, y_pred))
            print(model_name + '\'s best precision score:\n', precision_score(y_test, y_pred, zero_division=0))
            print(model_name + '\'s best recall score:\n', recall_score(y_test, y_pred, zero_division=0))
            print(model_name + '\'s best f1 score:\n', f1_score(y_test, y_pred, zero_division=0))
            print()

            viz_classification(grid_search_cv.best_estimator_, x_test, y_test, normalize='all',
                               mode=['cfm', 'roc', 'prc'])

            to_push = {'Dataset No.': idx, model_name: best_score_dict}
            best_score_dict_list.append(to_push)

    return best_score_dict_list


def findBestCluster(X, y):
    '''
    find set of best parameters of each cluster algorithm according to silhouette score and purity.

    :param X: list of data scaled with each scaler
    :param y: target feature

    return
    list[{kmeans}, {em}, {clarans}, {affinity propagation}]
    each dict contains best silhouette score, params, idx and best purity, params, idx.
    '''

    best_kmeans = {'silhouette score': None, 'silhouette param': None, 'silhouette idx': None,
                   'purity': None, 'purity param': None, 'purity idx': None}
    best_em = {'silhouette score': None, 'silhouette param': None, 'silhouette idx': None,
               'purity': None, 'purity param': None, 'purity idx': None}
    best_dbscan = {'silhouette score': None, 'silhouette param': None, 'silhouette idx': None,
               'purity': None, 'purity param': None, 'purity idx': None}
    best_meanShift = {'silhouette score': None, 'silhouette param': None, 'silhouette idx': None,
               'purity': None, 'purity param': None, 'purity idx': None}

    #models = ['kmeans', 'em', 'dbscan', 'meanShift']
    models = ['kmeans']
    kmeans_param = {'n_clusters': [3, 4, 7, 10, 30, 50, 80, 100, 200], 'algorithm': ['full', 'elkan']}
    em_param = {'n_components': [3, 5, 10, 30, 50, 80, 100, 200], 'covariance_type': ['full', 'tied'], 'tol': [1e-2, 1e-3, 1e-4]}
    dbscan_param = {'eps': [0.2, 0.3, 0.5, 0.7, 0.9, 1.2], 'min_samples': [3, 4, 5, 7, 10, 30, 50, 100]}
    meanShift_param = {'bandwidth': [0.5, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]}

    for model_name in models:

        # KMeans
        if model_name == 'kmeans':
            model = KMeans()
            idx = 0
            for x in X:
                cluster = GridSearchCV(estimator=model, param_grid=kmeans_param, cv=3, scoring='accuracy')
                cluster.fit(x, y[idx])
                predict = cluster.predict(x)

                silhouette = silhouette_score(x, predict, metric='euclidean')
                if best_kmeans['silhouette score'] == None or best_kmeans['silhouette score'] < silhouette:
                    best_kmeans['silhouette score'] = silhouette
                    best_kmeans['silhouette param'] = cluster.best_params_
                    best_kmeans['silhouette idx'] = idx

                p = contingency_matrix(y[idx], predict)
                purity = np.sum(np.amax(p, axis=0)) / np.sum(p)
                if best_kmeans['purity'] == None or best_kmeans['purity'] < purity:
                    best_kmeans['purity'] = purity
                    best_kmeans['purity param'] = cluster.best_params_
                    best_kmeans['purity idx'] = idx

                idx += 1

        # Gaussian Mixture(EM)
        elif model_name == 'em':
            model = GaussianMixture()
            idx = 0
            for x in X:
                cluster = GridSearchCV(estimator=model, param_grid=em_param, cv=3, scoring='accuracy')
                cluster.fit(x, y[idx])
                predict = cluster.predict(x)

                silhouette = silhouette_score(x, predict, metric='euclidean')
                if best_em['silhouette score'] == None or best_em['silhouette score'] < silhouette:
                    best_em['silhouette score'] = silhouette
                    best_em['silhouette param'] = cluster.best_params_
                    best_em['silhouette idx'] = idx

                p = contingency_matrix(y[idx], predict)
                purity = np.sum(np.amax(p, axis=0)) / np.sum(p)
                if best_em['purity'] == None or best_em['purity'] < purity:
                    best_em['purity'] = purity
                    best_em['purity param'] = cluster.best_params_
                    best_em['purity idx'] = idx

                idx += 1

        # DBSCAN
        elif model_name == 'dbscan':
            idx = 0
            for x in enumerate(X):
                input = x[1].values.tolist()
                for eps_value in dbscan_param['eps']:
                    for minSamples in dbscan_param['min_samples']:
                        cluster = DBSCAN(eps=eps_value, min_samples=minSamples)
                        cluster.fit(input)
                        core_samples_mask = np.zeros_like(cluster.labels_, dtype=bool)
                        core_samples_mask[cluster.core_sample_indices_] = True
                        labels = cluster.labels_

                        silhouette = silhouette_score(input, labels)
                        if best_dbscan['silhouette score'] == None or best_dbscan['silhouette score'] < silhouette:
                            best_dbscan['silhouette score'] = silhouette
                            best_dbscan['silhouette param'] = {'eps': eps_value, 'min_samples': minSamples}
                            best_dbscan['silhouette idx'] = idx

                idx += 1

        # Mean Shift
        elif model_name == 'meanShift':
            model = MeanShift()
            idx = 0
            for x in X:
                cluster = GridSearchCV(estimator=model, param_grid=meanShift_param, cv=3, scoring='accuracy')
                cluster.fit(x, y[idx])
                predict = cluster.predict(x)

                silhouette = silhouette_score(x, predict, metric='euclidean')
                if best_meanShift['silhouette score'] == None or best_meanShift['silhouette score'] < silhouette:
                    best_meanShift['silhouette score'] = silhouette
                    best_meanShift['silhouette param'] = cluster.best_params_
                    best_meanShift['silhouette idx'] = idx

                p = contingency_matrix(y[idx], predict)
                purity = np.sum(np.amax(p, axis=0)) / np.sum(p)
                if best_meanShift['purity'] == None or best_meanShift['purity'] < purity:
                    best_meanShift['purity'] = purity
                    best_meanShift['purity param'] = cluster.best_params_
                    best_meanShift['purity idx'] = idx

                idx += 1

        '''
        elif model_name == 'clarans':
            idx = 0
            for x in enumerate(X):
                input = x[1].values.tolist()
                for num_cluster in clarans_param['number_clusters']:
                    for num_local in clarans_param['numlocal']:
                        for max_nb in clarans_param['maxneighbor']:
                            cluster = clarans(input, num_cluster, num_local, max_nb)
                            cluster.process()

                            predict = np.zeros((len(input)))
                            i = 1
                            for ip in cluster.get_clusters():
                                for k in ip:
                                    predict[k] = i
                                i += 1

                            silhouette = silhouette_score(input, predict, metric='euclidean')
                            if best_clarans['silhouette score'] == None or best_clarans['silhouette score'] < silhouette:
                                best_clarans['silhouette score'] = silhouette
                                best_clarans['silhouette param'] = {'number_clusters': num_cluster, 'numlocal': num_local, 'maxneighbor': max_nb}
                                best_clarans['silhouette idx'] = idx
                idx += 1
        elif model_name == 'affinity':
            model = AffinityPropagation()
            idx = 0
            for x in X:
                cluster = GridSearchCV(estimator=model, param_grid=affinity_param, cv=3, scoring='accuracy')
                cluster.fit(x, y)
                predict = cluster.predict(x)

                silhouette = silhouette_score(x, predict, metric='euclidean')
                if best_affinity['silhouette score'] == None or best_affinity['silhouette score'] < silhouette:
                    best_affinity['silhouette score'] = silhouette
                    best_affinity['silhouette param'] = cluster.best_params_
                    best_affinity['silhouette idx'] = idx

                p = contingency_matrix(y, predict)
                purity = np.sum(np.amax(p, axis=0)) / np.sum(p)
                if best_affinity['purity'] == None or best_affinity['purity'] < purity:
                    best_affinity['purity'] = purity
                    best_affinity['purity param'] = cluster.best_params_
                    best_affinity['purity idx'] = idx

                idx += 1
        '''

    result = [best_kmeans, best_em, best_dbscan, best_meanShift]
    return result


# Plot clustering results
def plotCluster(X, param_list):
    # Reduce number of predictor features using PCA
    x_pca = []
    for x in X:
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(x)
        x_pca.append(data_pca)

    for i in range(0,4):
        param = param_list[i]
        if param['silhouette score'] == None:
            return

        # KMeans
        if i == 0:
            # Plot the cluster with the highest silhouette score.
            data = x_pca[param['silhouette idx']]
            model = KMeans(n_clusters=param['silhouette param']['n_clusters'], algorithm=param['silhouette param']['algorithm'])
            label = model.fit_predict(data)

            u_labels = np.unique(label)
            centroids = np.array(model.cluster_centers_)
            for k in u_labels:
                plt.scatter(data[label==k, 0], data[label==k, 1], label=k)
            plt.scatter(centroids[:,0], centroids[:,1], s=80, marker='x', color='k')
            plt.title("KMeans\nBest Silhouette")
            plt.legend()
            plt.show()

            # Plot the cluster with the highest purity.
            data = x_pca[param['purity idx']]
            model = KMeans(n_clusters=param['purity param']['n_clusters'],
                           algorithm=param['purity param']['algorithm'])
            label = model.fit_predict(data)

            u_labels = np.unique(label)
            centroids = np.array(model.cluster_centers_)
            for k in u_labels:
                plt.scatter(data[label == k, 0], data[label == k, 1], label=k)
            plt.scatter(centroids[:, 0], centroids[:, 1], s=80, marker='x', color='k')
            plt.title("KMeans\nBest Purity")
            plt.legend()
            plt.show()
        
        # GaussianMixture(EM)
        elif i == 1:
            # Plot the cluster with the highest silhouette score.
            data = x_pca[param['silhouette idx']]
            model = GaussianMixture(n_components=param['silhouette param']['n_components'],
                                    covariance_type=param['silhouette param']['covariance_type'], tol=param['silhouette param']['tol'])
            label = model.fit_predict(data)

            u_labels = np.unique(label)
            for k in u_labels:
                plt.scatter(data[label == k, 0], data[label == k, 1], label=k)
            plt.title("EM\nBest Silhouette")
            plt.legend()
            plt.show()

            # # Plot the cluster with the highest purity.
            data = x_pca[param['purity idx']]
            model = GaussianMixture(n_components=param['purity param']['n_components'],
                                    covariance_type=param['purity param']['covariance_type'],
                                    tol=param['purity param']['tol'])
            label = model.fit_predict(data)

            u_labels = np.unique(label)
            for k in u_labels:
                plt.scatter(data[label == k, 0], data[label == k, 1], label=k)
            plt.title("EM\nBest Purity")
            plt.legend()
            plt.show()
        
        # DBSCAN
        elif i == 2:
            # Plot the cluster with the highest silhouette score.
            data = x_pca[param['silhouette idx']]
            model = DBSCAN(eps=param['silhouette param']['eps'], min_samples=param['silhouette param']['min_samples'],
                           algorithm=param['silhouette param']['algorithm']).fit(data)
            core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
            core_samples_mask[model.core_sample_indices_] = True
            labels = model.labels_

            u_labels = set(labels)
            colors = [plt.cm.Spectral(each) for each in np.linspace(0,1,len(u_labels))]
            for k, col in zip(u_labels, colors):
                if k==-1:
                    col = [0,0,0,1]
                class_member_mask = labels == k

                xy = data[class_member_mask & core_samples_mask]
                plt.plot(xy[:,0], xy[:,1], "o", markerfacecolor=tuple(col), markeredgecolor="k")

                xy = data[class_member_mask & ~core_samples_mask]
                plt.plot(xy[:,0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k")
            plt.title("DBSCAN\nBest Silhouette")
            plt.legend()
            plt.show()

        # MeanShift
        elif i == 3:
            # Plot the cluster with the highest silhouette score.
            data = x_pca[param['silhouette idx']]
            model = MeanShift(bandwidth=param['silhouette param']['bandwidth'])
            model.fit(data)
            labels = model.labels_
            cluster_centers = model.cluster_centers_

            u_labels = np.unique(labels)
            n_clusters_ = len(u_labels)

            colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
            for k, col in zip(range(n_clusters_), colors):
                my_members = labels == k
                cluster_center = cluster_centers[k]
                plt.plot(data[my_members, 0], data[my_members, 1], col+".")
                plt.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor=col, markeredgecolor="k")
            plt.title("MeanShift\nBest Silhouette")
            plt.legend()
            plt.show()

            # Plot the cluster with the highest purity.
            data = x_pca[param['purity idx']]
            model = MeanShift(bandwidth=param['silhouette param']['bandwidth'])
            model.fit(data)
            labels = model.labels_
            cluster_centers = model.cluster_centers_

            u_labels = np.unique(labels)
            n_clusters_ = len(u_labels)

            colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
            for k, col in zip(range(n_clusters_), colors):
                my_members = labels == k
                cluster_center = cluster_centers[k]
                plt.plot(data[my_members, 0], data[my_members, 1], col + ".")
                plt.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor=col, markeredgecolor="k")
            plt.title("MeanShift\nBest Purity")
            plt.legend()
            plt.show()


# Read the dataset
original = pd.read_csv('credit_dataset.csv', index_col=0)

print(original)
print(original.info())
print(original.describe())
print(original.isnull().sum())  # It is a clean dataset without missing values.

# Start dataset preprocessing
original.apply(pd.unique)

# Drop unuseful columns "ID" and "FLAG_MOBIL"
data = original.drop(["ID", "FLAG_MOBIL"], axis=1)

auto_ml(data)
