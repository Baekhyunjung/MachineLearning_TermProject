import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, plot_roc_curve, plot_confusion_matrix, \
    PrecisionRecallDisplay, classification_report, make_scorer, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def data_refining(df):
    print(df)
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())  # 0 means clean dataset without missing values.

    # Start dataset preprocessing
    df.apply(pd.unique)

    # Drop unuseful columns "ID" and "FLAG_MOBIL"
    df = df.drop(["ID", "FLAG_MOBIL"], axis=1)

    return df


def feature_selection(df):
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
    preprocessed_dataset_list_smote = []  # smoted data with Index: 0-Standard, 1-MinMax, 2-MaxAbs, 3-Robust scale
    preprocessed_target_smote = []  # target value of each smoted data
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
    k_best_fs = []
    print('SelectKBest')
    i = 0
    for x in preprocessed_dataset_list_smote:
        train_x, test_x, train_y, test_y = train_test_split(x, preprocessed_target_smote[i], test_size=0.2)
        model = SelectKBest(score_func=f_classif, k=4)
        k_best = model.fit(train_x, train_y)
        support = k_best.get_support()
        features = x.columns
        print(features[support])
        new_x = x.loc[:, features[support]]
        k_best_fs.append(new_x)
        i += 1

    # Feature selection - RFE
    rfe_fs = []
    print('RFE')
    i = 0
    for x in preprocessed_dataset_list_smote:
        train_x, test_x, train_y, test_y = train_test_split(x, preprocessed_target_smote[i], test_size=0.2)
        model = LogisticRegression()
        rfe = RFE(model, n_features_to_select=4)
        fit = rfe.fit(train_x, train_y)
        support = fit.get_support()
        features = x.columns
        print(features[support])
        new_x = x.loc[:, features[support]]
        rfe_fs.append(new_x)
        i += 1

    # Feature selection - SelectFromModel
    sfm_fs = []
    print('SelectFromModel')
    i = 0
    for x in preprocessed_dataset_list_smote:
        train_x, test_x, train_y, test_y = train_test_split(x, preprocessed_target_smote[i], test_size=0.2)
        selector = SelectFromModel(estimator=LogisticRegression()).fit(train_x, train_y)
        features = x.columns
        print(features[selector.get_support()])
        new_x = x.loc[:, features[selector.get_support()]]
        sfm_fs.append(new_x)
        i += 1

    return intuitive_fs, k_best_fs, rfe_fs, sfm_fs, preprocessed_target_smote


def auto_ml_classification(df, models, model_names, param_grid_1, param_grid_2, param_grid_3):
    # Feature selection
    intuitive_fs, k_best_fs, rfe_fs, sfm_fs, preprocessed_target_smote = feature_selection(df)

    # Classification
    find_best_classification(intuitive_fs, preprocessed_target_smote, models, model_names, param_grid_1, param_grid_2, param_grid_3)
    find_best_classification(k_best_fs, preprocessed_target_smote, models, model_names, param_grid_1, param_grid_2, param_grid_3)
    find_best_classification(rfe_fs, preprocessed_target_smote, models, model_names, param_grid_1, param_grid_2, param_grid_3)
    find_best_classification(sfm_fs, preprocessed_target_smote, models, model_names, param_grid_1, param_grid_2, param_grid_3)


def auto_ml_clustering(df, model_names, param_1, param_2, param_3, param_4):
    # Feature selection
    intuitive_fs, k_best_fs, rfe_fs, sfm_fs, preprocessed_target_smote = feature_selection(df)

    # Clustering
    intuitive_cluster = find_best_cluster(intuitive_fs, preprocessed_target_smote, model_names, param_1, param_2, param_3, param_4)
    k_best_cluster = find_best_cluster(k_best_fs, preprocessed_target_smote, model_names, param_1, param_2, param_3, param_4)
    rfe_cluster = find_best_cluster(rfe_fs, preprocessed_target_smote, model_names, param_1, param_2, param_3, param_4)
    sfm_cluster = find_best_cluster(sfm_fs, preprocessed_target_smote, model_names, param_1, param_2, param_3, param_4)

    plot_cluster(intuitive_fs, intuitive_cluster)
    plot_cluster(k_best_fs, k_best_cluster)
    plot_cluster(rfe_fs, rfe_cluster)
    plot_cluster(sfm_fs, sfm_cluster)


def encode_scale(df, numerical_feature_list, categorical_feature_list, target_name, scalers=None, encoders=None,
                 fill_nan=None, outliers=None):
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


def find_best_classification(x_list, y, models, model_names, param_grid_1, param_grid_2, param_grid_3):
    """
    find the set of best parameters among classification algorithms according to silhouette score and purity.
    :param x_list: list of data encoded and scaled with specified features
    :param y: target feature
    :param models: list of model classes
    :param model_names: list of model name strings
    :param param_grid_1, param_grid_2, param_grid_3: dictionary of model parameters grid
    return
    list[{decisionTree}, {logisticRegression}, {svm}]
    each dict contains best precision, recall, f1 score, params, idx.
    """
    # Parameter grids list
    param_grids = [param_grid_1, param_grid_2, param_grid_3]

    # Scoring criteria
    scorers = {'precision_score': make_scorer(precision_score),
               'recall_score': make_scorer(recall_score),
               'f1_score': make_scorer(f1_score)}
    best_score_dict_list = []

    for idx, data_x in enumerate(x_list):
        x_train, x_test, y_train, y_test = train_test_split(data_x, y[idx], shuffle=True, stratify=y[idx])
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


def viz_classification(model, x, y, normalize, mode=None):
    """
    :param model:
    :param x:
    :param y:
    :param normalize:
    :param mode: ROC curve, PRC, CAP curve and the confusion matrix.
    :return:
    """
    # mode - confusion_matrix
    if "cfm" in mode:
        plot = plot_confusion_matrix(model, x, y, normalize=normalize)
        plot.ax_.set_title("Confusion Matrix")

    if "roc" in mode:
        plot = plot_roc_curve(model, x, y)
        plot.ax_.set_title("ROC Curve")

    if "prc" in mode:
        plot = PrecisionRecallDisplay.from_estimator(model, x, y)
        plot.ax_.set_title("Precision-Recall curve")


def find_best_cluster(X, y, model_names, param_1, param_2, param_3, param_4):
    """
    find set of best parameters of each cluster algorithm according to silhouette score and purity.
    :param X: list of data scaled with each scaler
    :param y: target feature
    :param model_names: list of model name strings
    :param param_1, param_2, param_3, param_4: dictionary of model parameters grid
    return
    list[{model_1}, {model_2}, {model_3}, {model_4}]
    each dict contains its model's best silhouette score, params, idx and best purity, params, idx.
    """

    best_result_1 = {'silhouette score': None, 'silhouette param': None, 'silhouette idx': None,
                     'purity': None, 'purity param': None, 'purity idx': None}
    best_result_2 = {'silhouette score': None, 'silhouette param': None, 'silhouette idx': None,
                     'purity': None, 'purity param': None, 'purity idx': None}
    best_result_3 = {'silhouette score': None, 'silhouette param': None, 'silhouette idx': None,
                     'purity': None, 'purity param': None, 'purity idx': None}
    best_result_4 = {'silhouette score': None, 'silhouette param': None, 'silhouette idx': None,
                     'purity': None, 'purity param': None, 'purity idx': None}

    for model_name in model_names:
        # KMeans
        if model_name == 'KMeans':
            model = KMeans()
            idx = 0
            for x in X:
                cluster = GridSearchCV(estimator=model, param_grid=param_1, cv=3, scoring='accuracy')
                cluster.fit(x, y[idx])
                predict = cluster.predict(x)

                silhouette = silhouette_score(x, predict, metric='euclidean')
                if best_result_1['silhouette score'] is None or best_result_1['silhouette score'] < silhouette:
                    best_result_1['silhouette score'] = silhouette
                    best_result_1['silhouette param'] = cluster.best_params_
                    best_result_1['silhouette idx'] = idx

                p = contingency_matrix(y[idx], predict)
                purity = np.sum(np.amax(p, axis=0)) / np.sum(p)
                if best_result_1['purity'] is None or best_result_1['purity'] < purity:
                    best_result_1['purity'] = purity
                    best_result_1['purity param'] = cluster.best_params_
                    best_result_1['purity idx'] = idx

                idx += 1

        # Gaussian Mixture(EM)
        elif model_name == 'EM':
            model = GaussianMixture()
            idx = 0
            for x in X:
                cluster = GridSearchCV(estimator=model, param_grid=param_2, cv=3, scoring='accuracy')
                cluster.fit(x, y[idx])
                predict = cluster.predict(x)

                silhouette = silhouette_score(x, predict, metric='euclidean')
                if best_result_2['silhouette score'] is None or best_result_2['silhouette score'] < silhouette:
                    best_result_2['silhouette score'] = silhouette
                    best_result_2['silhouette param'] = cluster.best_params_
                    best_result_2['silhouette idx'] = idx

                p = contingency_matrix(y[idx], predict)
                purity = np.sum(np.amax(p, axis=0)) / np.sum(p)
                if best_result_2['purity'] is None or best_result_2['purity'] < purity:
                    best_result_2['purity'] = purity
                    best_result_2['purity param'] = cluster.best_params_
                    best_result_2['purity idx'] = idx

                idx += 1

        # DBSCAN
        elif model_name == 'DBSCAN':
            idx = 0
            for x in enumerate(X):
                input_list = x[1].values.tolist()
                for eps_value in param_3['eps']:
                    for minSamples in param_3['min_samples']:
                        cluster = DBSCAN(eps=eps_value, min_samples=minSamples)
                        cluster.fit(input_list)
                        core_samples_mask = np.zeros_like(cluster.labels_, dtype=bool)
                        core_samples_mask[cluster.core_sample_indices_] = True
                        labels = cluster.labels_

                        silhouette = silhouette_score(input_list, labels)
                        if best_result_3['silhouette score'] is None or best_result_3['silhouette score'] < silhouette:
                            best_result_3['silhouette score'] = silhouette
                            best_result_3['silhouette param'] = {'eps': eps_value, 'min_samples': minSamples}
                            best_result_3['silhouette idx'] = idx

                idx += 1

        # Mean Shift
        elif model_name == 'MeanShift':
            model = MeanShift()
            idx = 0
            for x in X:
                cluster = GridSearchCV(estimator=model, param_grid=param_4, cv=3, scoring='accuracy', n_jobs=4)
                cluster.fit(x, y[idx])
                predict = cluster.predict(x)

                silhouette = silhouette_score(x, predict, metric='euclidean')
                if best_result_4['silhouette score'] is None or best_result_4['silhouette score'] < silhouette:
                    best_result_4['silhouette score'] = silhouette
                    best_result_4['silhouette param'] = cluster.best_params_
                    best_result_4['silhouette idx'] = idx

                p = contingency_matrix(y[idx], predict)
                purity = np.sum(np.amax(p, axis=0)) / np.sum(p)
                if best_result_4['purity'] is None or best_result_4['purity'] < purity:
                    best_result_4['purity'] = purity
                    best_result_4['purity param'] = cluster.best_params_
                    best_result_4['purity idx'] = idx

                idx += 1

        print(model_name + "'s best silhouette score:", best_result_4['silhouette score'])
        print(model_name + "'s best silhouette parameters:", best_result_4['silhouette param'])
        print('Target Data index:', best_result_4['silhouette idx'])
        print()
        print(model_name + "'s best purity score:", best_result_4['purity'])
        print(model_name + "'s best parameters:", best_result_4['purity param'])
        print('Target Data index:', best_result_4['purity idx'])

    result = [best_result_1, best_result_2, best_result_3, best_result_4]

    return result


# Plot clustering results
def plot_cluster(X, param_list):
    # Reduce number of predictor features using PCA
    x_pca = []
    for x in X:
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(x)
        x_pca.append(data_pca)

    for i in range(0, 4):
        param = param_list[i]
        if param['silhouette score'] is None:
            return

        # KMeans
        if i == 0:
            # Plot the cluster with the highest silhouette score.
            data_pca = x_pca[param['silhouette idx']]
            model = KMeans(n_clusters=param['silhouette param']['n_clusters'],
                           algorithm=param['silhouette param']['algorithm'])
            label = model.fit_predict(data_pca)

            u_labels = np.unique(label)
            centroids = np.array(model.cluster_centers_)
            for k in u_labels:
                plt.scatter(data_pca[label == k, 0], data_pca[label == k, 1], label=k)
            plt.scatter(centroids[:, 0], centroids[:, 1], s=80, marker='x', color='k')
            plt.title("KMeans\nBest Silhouette")
            plt.legend()
            plt.show()

            # Plot the cluster with the highest purity.
            data_pca = x_pca[param['purity idx']]
            model = KMeans(n_clusters=param['purity param']['n_clusters'],
                           algorithm=param['purity param']['algorithm'])
            label = model.fit_predict(data_pca)

            u_labels = np.unique(label)
            centroids = np.array(model.cluster_centers_)
            for k in u_labels:
                plt.scatter(data_pca[label == k, 0], data_pca[label == k, 1], label=k)
            plt.scatter(centroids[:, 0], centroids[:, 1], s=80, marker='x', color='k')
            plt.title("KMeans\nBest Purity")
            plt.legend()
            plt.show()

        # GaussianMixture(EM)
        elif i == 1:
            # Plot the cluster with the highest silhouette score.
            data_pca = x_pca[param['silhouette idx']]
            model = GaussianMixture(n_components=param['silhouette param']['n_components'],
                                    covariance_type=param['silhouette param']['covariance_type'],
                                    tol=param['silhouette param']['tol'])
            label = model.fit_predict(data_pca)

            u_labels = np.unique(label)
            for k in u_labels:
                plt.scatter(data_pca[label == k, 0], data_pca[label == k, 1], label=k)
            plt.title("EM\nBest Silhouette")
            plt.legend()
            plt.show()

            # # Plot the cluster with the highest purity.
            data_pca = x_pca[param['purity idx']]
            model = GaussianMixture(n_components=param['purity param']['n_components'],
                                    covariance_type=param['purity param']['covariance_type'],
                                    tol=param['purity param']['tol'])
            label = model.fit_predict(data_pca)

            u_labels = np.unique(label)
            for k in u_labels:
                plt.scatter(data_pca[label == k, 0], data_pca[label == k, 1], label=k)
            plt.title("EM\nBest Purity")
            plt.legend()
            plt.show()

        # DBSCAN
        elif i == 2:
            # Plot the cluster with the highest silhouette score.
            data_pca = x_pca[param['silhouette idx']]
            model = DBSCAN(eps=param['silhouette param']['eps'], min_samples=param['silhouette param']['min_samples'],
                           algorithm=param['silhouette param']['algorithm']).fit(data_pca)
            core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
            core_samples_mask[model.core_sample_indices_] = True
            labels = model.labels_

            u_labels = set(labels)
            colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(u_labels))]
            for k, col in zip(u_labels, colors):
                if k == -1:
                    col = [0, 0, 0, 1]
                class_member_mask = labels == k

                xy = data_pca[class_member_mask & core_samples_mask]
                plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k")

                xy = data_pca[class_member_mask & ~core_samples_mask]
                plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k")
            plt.title("DBSCAN\nBest Silhouette")
            plt.legend()
            plt.show()

        # MeanShift
        elif i == 3:
            # Plot the cluster with the highest silhouette score.
            data_pca = x_pca[param['silhouette idx']]
            model = MeanShift(bandwidth=param['silhouette param']['bandwidth'])
            model.fit(data_pca)
            labels = model.labels_
            cluster_centers = model.cluster_centers_

            u_labels = np.unique(labels)
            n_clusters_ = len(u_labels)

            colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
            for k, col in zip(range(n_clusters_), colors):
                my_members = labels == k
                cluster_center = cluster_centers[k]
                plt.plot(data_pca[my_members, 0], data_pca[my_members, 1], col + ".")
                plt.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor=col, markeredgecolor="k")
            plt.title("MeanShift\nBest Silhouette")
            plt.legend()
            plt.show()

            # Plot the cluster with the highest purity.
            data_pca = x_pca[param['purity idx']]
            model = MeanShift(bandwidth=param['silhouette param']['bandwidth'])
            model.fit(data_pca)
            labels = model.labels_
            cluster_centers = model.cluster_centers_

            u_labels = np.unique(labels)
            n_clusters_ = len(u_labels)

            colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
            for k, col in zip(range(n_clusters_), colors):
                my_members = labels == k
                cluster_center = cluster_centers[k]
                plt.plot(data_pca[my_members, 0], data_pca[my_members, 1], col + ".")
                plt.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor=col, markeredgecolor="k")
            plt.title("MeanShift\nBest Purity")
            plt.legend()
            plt.show()


# Read the dataset
original = pd.read_csv('credit_dataset.csv', index_col=0)

# Auto data refining
data = data_refining(original)

# Auto classification step 1: Set the models for auto classification
models_list = [DecisionTreeClassifier(), LogisticRegression(), XGBClassifier()]
model_names_list = ['DecisionTreeClassifier', 'LogisticRegression', 'XGB']

# Auto classification step 2: Set the parameters for auto classification
decision_tree_param_grid = {'criterion': ['entropy', 'gini'], 'max_depth': [3, 4, 5],
                            'min_samples_split': [2, 4, 6], 'min_samples_leaf': [1, 3, 5]}
logistic_regression_param_grid = {'penalty': ['l2'], 'C': np.logspace(-4, 4, 20),
                                  'solver': ['newton-cg', 'lbfgs', 'liblinear']}
xgb_param_grid = {'n_estimators': [100, 300, 500, 1000], 'learning_rate': [0.01, 0.1, 1, 10], 'max_depth': [4, 8, 12]}

# Auto classification step 3: Call the auto classification function
auto_ml_classification(data, models_list, model_names_list,
                       decision_tree_param_grid, logistic_regression_param_grid, xgb_param_grid)

# Auto classification step 1: Set the models for auto clustering
model_names_list = ['KMeans', 'EM', 'DBSCAN', 'MeanShift']

# Auto classification step 2: Set the parameters for auto clustering
kmeans_param = {'n_clusters': [3, 4, 7, 10, 30, 50, 80, 100, 200], 'algorithm': ['full', 'elkan']}
em_param = {'n_components': [3, 5, 10, 30, 50, 80, 100, 200], 'covariance_type': ['full', 'tied'],
            'tol': [1e-2, 1e-3, 1e-4]}
dbscan_param = {'eps': [0.2, 0.3, 0.5, 0.7, 0.9, 1.2], 'min_samples': [3, 4, 5, 7, 10, 30, 50, 100]}
mean_shift_param = {'bandwidth': [0.5, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]}

# Auto classification step 3: Call the auto classification function
auto_ml_clustering(data, model_names_list, kmeans_param, em_param, dbscan_param, mean_shift_param)
