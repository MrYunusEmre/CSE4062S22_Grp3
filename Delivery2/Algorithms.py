import logging

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import mean, std
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix, \
    precision_recall_fscore_support
from sklearn.model_selection import train_test_split, cross_validate, KFold, cross_val_score, StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

PATH = "IssueTickets.ods"


class Algorithms:
    def __init__(self):
        logging.basicConfig(filename="DataHandler.log", filemode="w", level=logging.INFO)
        self.logger = logging.getLogger(f'{self.__class__.__name__}')
        logging.info(f'{self.__class__.__name__} is initialized')

        self.file_path = PATH

        self.get_x_y_from_dataset()

    def get_x_y_from_dataset(self):

        # features = "REPORTER","ISSUE_TYPE","PRIORITY","COMPNAME","WORKER","EMPLOYEE_TYPE","WORK_LOG","CREATION_DATE","RESOLUTION_DATE","ISSUE_CATEGORY"

        self.fields = ["REPORTER", "ISSUE_TYPE", "PRIORITY", "COMPNAME", "WORKER", "EMPLOYEE_TYPE", "WORK_LOG",
                       "ISSUE_CATEGORY"]

        data_frame = pd.read_excel(self.file_path, engine='odf', usecols=self.fields)
        dataset = data_frame.values

        self.X = dataset[:, :-1]
        self.y = dataset[:, -1]  # this will be the labels

        # format all fields as string
        self.X = self.X.astype(str)

    # prepare input data
    def prepare_inputs(self, X_train, X_test):
        oe = OrdinalEncoder()
        oe.fit(X_train)
        X_train_enc = oe.transform(X_train)
        X_test_enc = ""
        if (len(X_test) > 1):
            X_test_enc = oe.transform(X_test)
        return X_train_enc, X_test_enc

    # prepare target
    def prepare_targets(self, y_train, y_test):
        self.le = LabelEncoder()
        self.le.fit(y_train)
        y_train_enc = self.le.transform(y_train)
        y_test_enc = self.le.transform(y_test)
        return y_train_enc, y_test_enc

    def get_train_test_split(self):
        X_enc, _ = self.prepare_inputs(self.X, [])
        y_enc, _ = self.prepare_targets(self.y, [])

        X_train, X_test, y_train, y_test = train_test_split(X_enc, y_enc, test_size=0.33, random_state=1)

        # print(
        #     f"X_train shape: {X_train.shape} , X_test shape: {X_test.shape} , y_train shape: {y_train.shape} , y_test shape: {y_test.shape}")

        return X_train, X_test, y_train, y_test, y_enc, X_enc

    def KNN_algorithm(self):

        self.X_train, self.X_test, self.y_train, self.y_test, self.y_enc, self.X_enc = self.get_train_test_split()

        self.model = KNeighborsClassifier(n_neighbors=3)

        self.cross_validation(self.X_enc, self.y_enc)

        self.cmd_plot()

    def NaiveBayes_algorithm(self):

        self.X_train, self.X_test, self.y_train, self.y_test, self.y_enc, self.X_enc = self.get_train_test_split()

        # Create a Gaussian Classifier
        self.model = GaussianNB()

        self.cross_validation(self.X_enc, self.y_enc)

        self.cmd_plot()

    def SVM_algorithm(self):

        self.X_train, self.X_test, self.y_train, self.y_test, self.y_enc, self.X_enc = self.get_train_test_split()
        print("create svm classifier")
        # Create a svm Classifier
        self.model = svm.SVC(kernel='linear', probability=True)  # Linear Kernel
        print("cross validation")
        self.cross_validation(self.X_enc, self.y_enc)

        self.cmd_plot()


    def cross_validation(self, X, y, _cv=10):

        import warnings
        warnings.filterwarnings('ignore')

        skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)

        self.predictions = []

        self.accs = []
        self.reports = []
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
        self.X_trains_cv = []
        self.X_tests_cv = []
        self.y_trains_cv = []
        self.y_tests_cv = []
        index = 0
        for train_index, test_index in skfold.split(X, y):
            print(f"train index {train_index}\ntest index {test_index}")
            self.X_train_cv, self.X_test_cv = X[train_index], X[test_index]
            self.y_train_cv, self.y_test_cv = y[train_index], y[test_index]
            self.X_trains_cv.append(self.X_train_cv)
            self.X_tests_cv.append(self.X_test_cv)
            self.y_trains_cv.append(self.y_train_cv)
            self.y_tests_cv.append(self.y_test_cv)
            predictions, accuracy, metrics_report, nb_prf = self.train_test_model(self.X_train_cv, self.X_test_cv,
                                                                                  self.y_train_cv, self.y_test_cv)
            self.accs.append(accuracy)
            self.reports.append(metrics_report)
            self.precisions.append(nb_prf[0])
            self.recalls.append(nb_prf[1])
            self.f1_scores.append(nb_prf[2])

            self.roc_plot(self.X_test_cv, self.y_test_cv, index)
            index = index + 1

        print('mean accuracy: {:.2f}'.format(np.mean(self.accs)))
        print('mean precision: {:.2f}'.format(np.mean(self.precisions)))
        print('mean recall: {:.2f}'.format(np.mean(self.recalls)))
        print('mean f1 score: {:.2f}'.format(np.mean(self.f1_scores)))

        for report in self.reports:
            print("Fold number : ", self.reports.index(report))
            print(report)
            with open(
                    f"plots/{self.model.__class__.__name__}/{self.model.__class__.__name__}{self.reports.index(report)}.txt",
                    "w", encoding='UTF-8') as file:
                file.writelines(report)

        self.index_of_best_fold = self.f1_scores.index(max(self.f1_scores))

    def roc_plot(self, X_test, y_test, index):

        print(y_test)

        y_scores = self.model.predict_proba(X_test)

        classes = self.le.inverse_transform(self.model.classes_)

        fpr_list = []
        tpr_list = []
        roc_auc_list = []
        for i in self.model.classes_:
            fpr, tpr, thresholds = roc_curve(y_test, y_scores[:, i], pos_label=i)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            roc_auc_list.append(auc(fpr, tpr))

        plt.title('Receiver Operating Characteristic')
        i = 0
        for fpr, tpr in zip(fpr_list, tpr_list):
            plt.plot(fpr, tpr, label=f"ROC Curve of Class {classes[i]}")
            i += 1

        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title(f'ROC Curve of {self.model.__class__.__name__} -- fold {index}')

        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, f'plots/{self.model.__class__.__name__}')

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        plt.savefig(f"{results_dir}/{self.model.__class__.__name__}_roc_fold_{index}.png")
        plt.show()

    def cmd_plot(self):

        classes = self.le.inverse_transform(self.model.classes_)

        cm = confusion_matrix(self.y_tests_cv[self.index_of_best_fold], self.predictions[self.index_of_best_fold],
                              labels=self.model.classes_)
        display = ConfusionMatrixDisplay(confusion_matrix=cm,
                                         display_labels=classes)
        display.plot(xticks_rotation='vertical')

        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, f'plots/{self.model.__class__.__name__}')

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        plt.savefig(f"{results_dir}/{self.model.__class__.__name__}_confusion_matrix.png")
        plt.show()

# endregion

if __name__ == "__main__":
     algo = Algorithms()
     algo.KNN_algorithm()
    # algo1 = Algorithms()
    # algo1.NaiveBayes_algorithm()
    #algo2 = Algorithms()
    #algo2.SVM_algorithm()
    # algo3 = Algorithms()
    # algo3.ANN_algorithm()
    # algo4 = Algorithms()
    # algo4.DecisionTrees_algorithm()
    # algo5 = Algorithms()
    # algo5.RandomForest_algorithm()
    # algo6 = Algorithms()
    # algo6.SGD_algorithm()
