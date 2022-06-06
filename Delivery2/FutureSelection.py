import logging

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, RFECV, SelectFromModel, mutual_info_classif, \
    f_classif

from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PATH = "IssueTickets.ods"

class FutureSelection:
    def __init__(self):
        logging.basicConfig(filename="DataHandler.log", filemode="w", level=logging.INFO)
        self.logger = logging.getLogger(f'{self.__class__.__name__}')
        logging.info(f'{self.__class__.__name__} is initialized')

        self.file_path = PATH

        self.get_x_y_from_dataset()

    def get_x_y_from_dataset(self):

        self.fields = ["REPORTER","ISSUE_TYPE","PRIORITY","COMPNAME","WORKER","EMPLOYEE_TYPE","WORK_LOG","ISSUE_CATEGORY"]

        data_frame = pd.read_excel(self.file_path, engine='odf', usecols=self.fields)
        dataset = data_frame.values
        updated_dataset = []

        # #To place label at the end of the list
        # for data_list in dataset:
        #     data_list[7], data_list[9] = data_list[9],data_list[7]
        #     updated_dataset.append(data_list.tolist())

        self.X = dataset[:, :-1]
        self.y = dataset[:, -1] # this will be the labels

        # format all fields as string
        self.X = self.X.astype(str)


    # prepare input data
    def prepare_inputs(self,X_train,X_test):

        oe = OrdinalEncoder()
        oe.fit(X_train)
        X_train_enc = oe.transform(X_train)
        X_test_enc = ""
        if (len(X_test) > 1):
            X_test_enc = oe.transform(X_test)
        return X_train_enc, X_test_enc


    # prepare target
    def prepare_targets(self,y_train, y_test):
        le = LabelEncoder()
        le.fit(y_train)
        y_train_enc = le.transform(y_train)
        y_test_enc = le.transform(y_test)
        return y_train_enc, y_test_enc

    def remove_low_variance_features(self): # this method does not have any predict method. It just has fit method.

        # prepare input data
        X_enc , _ = self.prepare_inputs(self.X , [])
        print(X_enc.shape)
        sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        print(sel.fit(X_enc))

        print(sel.transform(X_enc))

        cols = sel.get_support(indices=1)
        col_names = [self.fields[col] for col in cols]

        print(f"col names = {col_names}")

    def apply_recursive_feature_selection(self): # uzun suruyorr
        X_enc, _ = self.prepare_inputs(self.X, [])

        m = RFECV(RandomForestClassifier(), scoring='accuracy')
        print(m.fit(X_enc, self.y))
        print(m.score(X_enc, self.y))


        cols = m.get_support(indices=1)
        col_names = [self.fields[col] for col in cols]

        print(f"col names = {col_names}")


    def apply_select_from_model(self):
        X_enc, _ = self.prepare_inputs(self.X, [])

        m = SelectFromModel(LinearSVC(C=0.01, penalty='l1', dual=False))

        print(m.fit(X_enc, self.y))

        print(f"before future selection : {X_enc.shape} , after future selection : {m.transform(X_enc).shape}")

        print(m.transform(X_enc))


        cols = m.get_support(indices=1)
        col_names = [self.fields[col] for col in cols]

        print(f"col names = {col_names}")


    def get_train_test_split(self):
        X_enc, _ = self.prepare_inputs(self.X, [])
        y_enc, _ = self.prepare_targets(self.y, [])

        X_train, X_test, y_train, y_test = train_test_split(X_enc, y_enc, test_size=0.33, random_state=1)

        print(
            f"X_train shape: {X_train.shape} , X_test shape: {X_test.shape} , y_train shape: {y_train.shape} , y_test shape: {y_test.shape}")

        return X_train, X_test, y_train, y_test


    def apply_pipeline_method(self):
        X_train, X_test, y_train, y_test = self.get_train_test_split()

        clf = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
        print(f"x_train_enc shape : f{X_train.shape} , y_train_enc shape : {y_train.shape}")
        print(clf.fit(X_train, y_train))
        print(f"x_test_enc shape : f{X_test.shape} , y_train_enc shape : {y_test.shape}")
        print(clf.score(X_test,y_test))

    #onemli olan metod..........
    def apply_univariate_feature_selection(self,selection_method=chi2): # this method does not have any predict method. It just has fit method.

        #chi2, f_classif, mutual_info_classif

        X_enc, _ = self.prepare_inputs(self.X, [])
        y_enc, _ = self.prepare_targets(self.y, [])

        sel = SelectKBest(selection_method, k='all')
        print(sel.fit(X_enc, y_enc))

        print(f"before future selection : {X_enc.shape} , after future selection : {sel.transform(X_enc).shape}")
        cols = sel.get_support(indices=1)
        col_names = [self.fields[col] for col in cols]

        print(f"scores : {sel.scores_}")

        print(f"col names = {col_names}")

        # plot the scores
        plt.bar([self.fields[i] for i in range(len(sel.scores_))], sel.scores_)
        plt.xticks(rotation=15,ha="right")

        plt.savefig(f"plots/feature_selection_{selection_method.__name__}.png")
        plt.show()


if __name__ == '__main__':
    futureSelection = FutureSelection()

    #futureSelection.remove_low_variance_features()
    #futureSelection.apply_recursive_feature_selection()
    #futureSelection.apply_select_from_model()
    #futureSelection.apply_pipeline_method()

    # For classification: chi2, f_classif, mutual_info_classif --- f_classif = ANOVA F-value
    # For regression: f_regression, mutual_info_regression
    futureSelection.apply_univariate_feature_selection(f_classif)


