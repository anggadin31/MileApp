import pandas as pd
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


class MileAppML:
    def __init__(self):
        with open('data-sample.json') as file:
            data = json.load(file)
        self.data = pd.json_normalize(data)
        self.epoch = 1
        self.batch = 128

    def __call__(self, *args, **kwargs):
        self.run()

    def run(self):
        try:
            x_train, x_test, y_train, y_test, dummy_column = self.preprocessing()
            model = self.train_model(x_train, y_train)
            class_report = self.test_model(model, x_test, y_test)
            print(class_report)
            self.data = self.undummify(self.data)
            self.data = self.data[
                ['taskCreatedTime', 'taskAssignedTo', 'branchDestination', 'receiverCity', 'taskStatusLabel']]
        except Exception as ex:
            error = ex
            raise Exception(ex)

    def undummify(self, df, prefix_sep="_"):
        cols2collapse = {
            item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
        }
        series_list = []
        for col, needs_to_collapse in cols2collapse.items():
            if needs_to_collapse:
                undummified = (
                    df.filter(like=col)
                    .idxmax(axis=1)
                    .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                    .rename(col)
                )
                series_list.append(undummified)
            else:
                series_list.append(df[col])
        undummified_df = pd.concat(series_list, axis=1)
        return undummified_df

    def preprocessing(self):
        try:
            self.data['taskCreatedTime'] = self.data['taskCreatedTime'].str.rsplit(' ', n=1).str.get(0)
            self.data['taskCreatedTime'] = pd.to_datetime(self.data['taskCreatedTime'])
            self.data = self.data.rename(columns={'UserVar.branch_dest': 'branchDestination',
                                                  'UserVar.receiver_city': 'receiverCity',
                                                  'UserVar.taskStatusLabel': 'taskStatusLabel'})
            self.data['taskStatusLabel'] = self.data['taskStatusLabel'].replace(['Failed', 'Success'], [0, 1])
            self.data = self.data[self.data['taskStatusLabel'].notna()]
            self.data['taskStatusLabel'] = self.data['taskStatusLabel'].astype(int)
            self.data['taskCreatedTime_day'] = self.data['taskCreatedTime'].dt.day_name()
            self.data = self.data[
                ['taskCreatedTime_day', 'taskAssignedTo', 'branchDestination', 'receiverCity',
                 'taskStatusLabel']]
            self.data = pd.get_dummies(self.data, sparse=False,
                                       columns=self.data.select_dtypes(include='object').columns)
            dummy_column = list(self.data.columns)
            boolean_column = self.data.select_dtypes(include='bool')
            for column in boolean_column:
                self.data[column] = self.data[column].astype(int)
            x = self.data.drop('taskStatusLabel', axis=1).values
            y = self.data['taskStatusLabel'].values

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
            return x_train, x_test, y_train, y_test, dummy_column

        except Exception as ex:
            raise Exception(ex)

    def train_model(self, x_train, y_train):
        model = DecisionTreeClassifier()
        model = model.fit(x_train, y_train)
        return model

    def test_model(self, model, x_test, y_test):
        prediction = model.predict(x_test)
        class_report = classification_report(y_test, prediction)
        return class_report
