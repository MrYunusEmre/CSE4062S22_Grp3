from DataHandler import DataHandler
import matplotlib.pyplot as plt


def plot_reporter_issue_number_result(dataHandler):
    reporter_list = list(dataHandler.reporter_dict.keys())
    issue_list = list(dataHandler.reporter_dict.values())

    plt.figure(figsize=(15, 10))
    plt.barh(reporter_list, issue_list)
    plt.xticks(rotation=45)
    plt.xlabel('Issue Counts')
    plt.ylabel('Reporters')
    plt.title("Reporters VS Number of issues opened")
    plt.figtext(0.5, 0.01, f"Reporter Count : {len(reporter_list)}", ha="center", va="center", fontsize=18,
                bbox={"facecolor": "orange", "alpha": 0.5})
    plt.savefig('plots/reporter_issue_count.png')


def plot_issue_type_number_result(dataHandler):
    issuetype_list = list(dataHandler.issuetype_dict.keys())
    issuttype_count_list = list(dataHandler.issuetype_dict.values())

    plt.figure(figsize=(15, 10))
    plt.barh(issuetype_list, issuttype_count_list)
    plt.xticks(rotation=45)
    plt.xlabel('Issue Counts')
    plt.ylabel('Issue Types')
    plt.title("Issue Type Counts")
    plt.figtext(0.5, 0.01, f"Issue Types Count : {len(issuetype_list)}", ha="center", va="center", fontsize=18,
                bbox={"facecolor": "orange", "alpha": 0.5})
    plt.savefig('plots/issuetype_count.png')


def plot_issue_priority_number_result(dataHandler):
    issuetypepriority_list = list(dataHandler.issuetype_priority_dict.keys())
    issuttypepriority_count_list = list(dataHandler.issuetype_priority_dict.values())

    plt.figure(figsize=(15, 10))
    plt.barh(issuetypepriority_list, issuttypepriority_count_list)
    plt.xticks(rotation=45)
    plt.xlabel('Issue Types - Priority Counts')
    plt.ylabel('Issue Types - priority')
    plt.title("Issue Type - Priority Counts")
    plt.figtext(0.5, 0.01, f"Issue Type - Priority Count : {len(issuetypepriority_list)}", ha="center", va="center",
                fontsize=14, bbox={"facecolor": "orange", "alpha": 0.5})
    plt.savefig('plots/issuetypepriority_count.png')


def plot_comp_issue_count_result(dataHandler):
    comp_list = list(dataHandler.compname_dict.keys())
    comp_issue_count_list = list(dataHandler.compname_dict.values())

    plt.figure(figsize=(15, 10))
    plt.barh(comp_list, comp_issue_count_list)
    plt.xticks(rotation=45)
    plt.xlabel('Issue Counts')
    plt.ylabel('Companies')
    plt.title("Company - Opened Issue Counts")
    plt.figtext(0.5, 0.01, f"Company Count : {len(comp_list)}", ha="center", va="center", fontsize=14,
                bbox={"facecolor": "orange", "alpha": 0.5})
    plt.savefig('plots/company_openedissue_count.png')


def plot_worker_count_result(dataHandler):
    worker_list = list(dataHandler.worker_dict.keys())
    worker_list_issue_count_list = list(dataHandler.worker_dict.values())

    plt.figure(figsize=(15, 10))
    plt.barh(worker_list, worker_list_issue_count_list)
    plt.xticks(rotation=45)
    plt.xlabel('Issue Counts')
    plt.ylabel('Workers')
    plt.title("Worker - Issue")
    plt.figtext(0.5, 0.01, f"Worker Count : {len(worker_list)}", ha="center", va="center", fontsize=14,
                bbox={"facecolor": "orange", "alpha": 0.5})
    plt.savefig('plots/worker_issue_count.png')


def plot_employeetype_count_result(dataHandler):
    employeetype_list = list(dataHandler.employeetype_dict.keys())
    employeetype_list_count_list = list(dataHandler.employeetype_dict.values())

    plt.figure(figsize=(15, 10))
    plt.barh(employeetype_list, employeetype_list_count_list)
    plt.xticks(rotation=45)
    plt.xlabel('Counts')
    plt.ylabel('Employee Types')
    plt.title("Employee Types")
    plt.figtext(0.5, 0.01, f"Employee Type Count : {len(employeetype_list)}", ha="center", va="center", fontsize=14,
                bbox={"facecolor": "orange", "alpha": 0.5})
    plt.savefig('plots/employeetype_count.png')


def plot_employeetype_issue_count_result(dataHandler):
    employeetype_issue_list = list(dataHandler.employeetype_issue_dict.keys())
    employeetype_issue_count_list = list(dataHandler.employeetype_issue_dict.values())

    plt.figure(figsize=(15, 10))
    plt.barh(employeetype_issue_list, employeetype_issue_count_list)
    plt.xticks(rotation=45)
    plt.xlabel('Issue Counts')
    plt.ylabel('Employee Types')
    plt.title("Issue - Employee Type")
    plt.figtext(0.5, 0.01, f"Employee Type - Issue Count : {len(employeetype_issue_list)}", ha="center", va="center",
                fontsize=14, bbox={"facecolor": "orange", "alpha": 0.5})
    plt.savefig('plots/employeetype_issue_count.png')


def plot_work_issuetype_result(dataHandler):
    work_issuetype_list = list(dataHandler.work_issuetype_dict.keys())
    work_issuetype_count_list = list(dataHandler.work_issuetype_dict.values())

    plt.figure(figsize=(15, 10))
    plt.barh(work_issuetype_list, work_issuetype_count_list)
    plt.xticks(rotation=45)
    plt.xlabel('Worked Minutes')
    plt.ylabel('Issue Types')
    plt.title("Issue Type - Worked Time")
    plt.figtext(0.5, 0.01, f"Issue Type - Work : {len(work_issuetype_list)}", ha="center", va="center", fontsize=14,
                bbox={"facecolor": "orange", "alpha": 0.5})
    plt.savefig('plots/issuetype_work_count.png')


def plot_work_issuecategory_result(dataHandler):
    work_issuecategory_list = list(dataHandler.work_issuecategory_dict.keys())
    work_issuecategory_count_list = list(dataHandler.work_issuecategory_dict.values())

    plt.figure(figsize=(15, 10))
    plt.barh(work_issuecategory_list, work_issuecategory_count_list)
    plt.xticks(rotation=45)
    plt.xlabel('Worked Minutes')
    plt.ylabel('Issue Categories')
    plt.title("Issue Category - Worked Time")
    # Total kategori sayisini verir
    plt.figtext(0.5, 0.01, f"Issue Category - Work : {len(work_issuecategory_list)}", ha="center", va="center",
                fontsize=14, bbox={"facecolor": "orange", "alpha": 0.5})
    plt.savefig('plots/issuecategory_work_count.png')


def plot_issue_category_result(dataHandler):
    issuecategory_list = list(dataHandler.issuecategory_dict.keys())
    issuecategory_count_list = list(dataHandler.issuecategory_dict.values())

    plt.figure(figsize=(15, 10))
    plt.barh(issuecategory_list, issuecategory_count_list)
    plt.xticks(rotation=45)
    plt.xlabel('Counts')
    plt.ylabel('Issue Categories')
    plt.title("Issue Category - Counts")
    # Total kategori sayisini verir
    plt.figtext(0.5, 0.01, f"Issue Category Count : {len(issuecategory_list)}", ha="center", va="center", fontsize=14,
                bbox={"facecolor": "orange", "alpha": 0.5})
    plt.savefig('plots/issuecategory_count.png')


def plot_issue_opened_day_result(dataHandler):
    issueopenedday_list = list(dataHandler.issue_openingday_dict.keys())
    issueopenedday_count_list = list(dataHandler.issue_openingday_dict.values())

    plt.figure(figsize=(15, 10))
    plt.barh(issueopenedday_list, issueopenedday_count_list)
    plt.xticks(rotation=45)
    plt.xlabel('Issue Opened')
    plt.ylabel('Day Of Month')
    plt.title("Issue Count - Day")
    # Yilin kac gununde issue acildiginin sayisini verir
    plt.figtext(0.5, 0.01, f"Issue Opened Day Count : {len(issueopenedday_list)}", ha="center", va="center",
                fontsize=14, bbox={"facecolor": "orange", "alpha": 0.5})
    plt.savefig('plots/issueopenedday_count.png')


def plot_issue_opened_month_result(dataHandler):
    issueopenedmonth_list = list(dataHandler.issue_openingmonth_dict.keys())
    issueopeneddaymonth_count_list = list(dataHandler.issue_openingmonth_dict.values())

    plt.figure(figsize=(15, 10))
    plt.barh(issueopenedmonth_list, issueopeneddaymonth_count_list)
    plt.xticks(rotation=45)
    plt.xlabel('Issue Opened')
    plt.ylabel('Month of Year')
    plt.title("Issue Count - Month")
    # Yilin kac ayinda issue acildiginin sayisini verir
    plt.figtext(0.5, 0.01, f"Issue Opened Month Count : {len(issueopenedmonth_list)}", ha="center", va="center",
                fontsize=14, bbox={"facecolor": "orange", "alpha": 0.5})
    plt.savefig('plots/issueopenedmonth_count.png')


if __name__ == '__main__':
    dataHandler = DataHandler()

    print(f"Total row number : {dataHandler.total_row_number}")
    print(f"Total row number : {dataHandler.total_column_number}")

    plot_reporter_issue_number_result(dataHandler)
    plot_issue_type_number_result(dataHandler)
    plot_issue_priority_number_result(dataHandler)
    plot_comp_issue_count_result(dataHandler)
    plot_worker_count_result(dataHandler)
    plot_employeetype_count_result(dataHandler)
    plot_employeetype_issue_count_result(dataHandler)
    plot_work_issuetype_result(dataHandler)
    plot_work_issuecategory_result(dataHandler)
    plot_issue_category_result(dataHandler)
    plot_issue_opened_day_result(dataHandler)
    plot_issue_opened_month_result(dataHandler)
