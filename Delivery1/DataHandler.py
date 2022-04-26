import logging
import pandas as pd


class DataHandler:

    def __init__(self):
        logging.basicConfig(filename="DataHandler.log",filemode="w",level=logging.INFO)
        self.logger = logging.getLogger(f'{self.__class__.__name__}')
        logging.info(f'{self.__class__.__name__} is initialized')
        self.file_path = "IssueTickets.ods"
        self.read_dataset()

    def read_dataset(self):
        self.logger.info(f"Dataset file {self.file_path} is started reading.")
        data_frame = pd.read_excel(self.file_path, engine='odf')  # pip install odfpy

        self.total_row_number = data_frame.shape[0]  # gives total number of rows in dataset
        self.total_column_number = data_frame.shape[1]  # gives total number of columns in dataset

        self.extract_features(data_frame)

    def extract_features(self, df):
        self.logger.info(f"Features are started extracting.")
        self.reporter_dict = {}  # to keep the number of reporters and total number of issues opened by reporter
        self.issuetype_dict = {}  # to keep the number of issue types and total number of issue types opened
        self.issuetype_priority_dict = {}  # to keep the number of issue types and priority of issues
        self.compname_dict = {}  # to keep the number of companies opened an issue
        self.worker_dict = {}  # to keep the number of workers working on an issue
        self.employeetype_dict = {}  # to keep the number of employees types
        self.employeetype_issue_dict = {}  # to keep the number of employees and they worked on issue types
        self.work_issuetype_dict = {}  # to keep the total minutes worked on issuetype
        self.work_issuecategory_dict = {}  # to keep the total minutes worked on issue category
        self.issuecategory_dict = {}  # to keep the number of issue categories
        self.issue_openingday_dict = {} # to keep the number of days issue opened
        self.issue_openingmonth_dict = {} # to keep the number of days issue opened

        for i, row in df.iterrows():  # i = row number , row = item
            self.reporter_dict = self.update_key_by_one(self.reporter_dict, row["REPORTER"])
            self.issuetype_dict = self.update_key_by_one(self.issuetype_dict, row["ISSUE_TYPE"])
            self.issuetype_priority_dict = self.update_key_by_one(self.issuetype_priority_dict,
                                                             row["ISSUE_TYPE"] + "_" + row["PRIORITY"])
            self.compname_dict = self.update_key_by_one(self.compname_dict, row["COMPNAME"])
            self.worker_dict = self.update_key_by_one(self.worker_dict, row["WORKER"])
            self.employeetype_dict = self.update_key_by_one(self.employeetype_dict, row["EMPLOYEE_TYPE"])
            self.employeetype_issue_dict = self.update_key_by_one(self.employeetype_issue_dict,
                                                             row["EMPLOYEE_TYPE"] + "_" + row["ISSUE_TYPE"])
            self.work_issuetype_dict = self.update_key_by_value(self.work_issuetype_dict, row["ISSUE_TYPE"], row["WORK_LOG"])
            self.work_issuecategory_dict = self.update_key_by_value(self.work_issuecategory_dict, row["ISSUE_CATEGORY"],
                                                               row["WORK_LOG"])
            self.issuecategory_dict = self.update_key_by_one(self.issuecategory_dict, row["ISSUE_CATEGORY"])
            self.issue_openingday_dict = self.update_key_by_one(self.issue_openingday_dict, row["CREATION_DATE"].strftime("%d"))
            self.issue_openingmonth_dict = self.update_key_by_one(self.issue_openingmonth_dict, row["CREATION_DATE"].strftime("%m"))

    def update_key_by_one(self, dict, key):  # to update the value of key in dict
        if key in dict.keys():
            dict.update({key: dict[key] + 1})
        else:
            dict[key] = 1
        return dict

    def update_key_by_value(self, dict, key, value):  # to update the value of key in dict
        if key in dict.keys():
            dict.update({key: dict[key] + value})
        else:
            dict[key] = value
        return dict






