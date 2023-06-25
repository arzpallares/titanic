# Core imports
import numpy as np
import pandas as pd

def fix_columns(data: pd.DataFrame) -> pd.DataFrame:
       """
       Runs the data cleaning and transformation steps
       """

       """
       Since our objective is to predict the survival of a passanger, the feature 'Survived'
       will be the target feature to be predicted (y).

       Knowing the target, the next step is to learn it's correlation with the other features.
       This is done using the 'corr' function, however it only works with numerical features.

       Before evaluating the correlation it is necesary to apply transformation to the categorical
       features and turn them into numerical ones.

       To implement the transformations without affecting the other features, the categorical features
       will be moved to a different variable.
       """
       cat_data = data[['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']].copy()
       data.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

       # print(cat_data.head(10))
       """
       The features Name, Ticket & Cabin aren't suitable for transformation into a numerical, so discard them 
       """
       cat_data.drop(['Name', 'Ticket', 'Cabin'], axis=1)

       """
       There are two main ways to convert a categorical feature into a numerical one,
       1- Use the 'get_dummies' function and create a binary feature for each possible result.
       2- Create a custom function and apply it to the feature.
       """

       data['isFemale'] = cat_data['Sex'].apply(lambda x: 1 if x == 'female' else 0)
       
       data = data.join(pd.get_dummies(cat_data['Embarked'])\
                     .rename(columns={'S': 'IsS', 'C': 'IsC'})\
                     .drop(['Q'], axis=1))

       """
       Output:
       Survived    891.0
       Pclass      891.0
       Age         714.0
       SibSp       891.0
       Parch       891.0
       Fare        891.0
       Embarked    891.0
       isFemale    891.0
       -------------------------------------------------------------------------
       the count of values for 'Age' is lower than the other features
       This, indicates the presence of empty values in the column.

       This can be solved by droping the instances without age or set a new value for it.
       Since there are many instances, droping them could cause a lose of valuable data.
       """
       # Updating Age data
       fem_mean = data[data['isFemale'] == 1]['Age'].mean()
       male_mean = data[data['isFemale'] == 0]['Age'].mean()

       data['Age'] = data.apply(
              lambda x: round(fem_mean, 2) if ['isFemale'] == 1 and np.isnan(x['Age']) == True else \
              round(male_mean, 2) if np.isnan(x['Age']) == True else x['Age'], axis=1)

       # Binning the Age
       age_groups = pd.cut(data['Age'], 
              bins=[0.0, 10.0, 18.0, 30.0, 50.0, 65.0, 100.0],
              labels=False)

       data['GroupAge'] = age_groups.apply(lambda x: x if np.isnan(x) == False  else 50.0) 

       # Combining SibSp and Parch
       data['HasFam'] = data['SibSp'] + data['Parch']

       data['femFam'] = data.apply(lambda x: 1 if x['isFemale'] == True and x['HasFam'] > 0 else 0, axis=1)

       data['femPclass'] = data.apply(lambda x: x['isFemale']/x['Pclass'], axis=1)

       return data.drop(['Age', 'SibSp', 'Parch', 'HasFam'], axis=1)


def fix_dtypes(data: pd.DataFrame) -> pd.DataFrame:
       data.describe(include='all')
       breakpoint()

if __name__ == '__main__':
       # Load the datasets into Dataframes
       train_data = pd.read_csv('data/train-titanic.csv', index_col='PassengerId')
       test_data = pd.read_csv('data/test-titanic.csv', index_col='PassengerId')

       data = pd.get_dummies(train_data['Embarked']).rename(columns={'S': 'IsS', 'C': 'IsC'}).drop(['Q'], axis=1)