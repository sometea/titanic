#!/usr/bin/env python

import pandas
import numpy

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.decomposition import PCA

def arrayFromCsv(filename, withAges=True):
    """ read in the csv, clean up the data, and return a numpy array """
    df = pandas.read_csv(filename, header=0)

    # convert Sex into 0 for female, 1 for male
    df['NumSex'] = df['Sex'].map({ 'female': 0, 'male': 1} ).astype(int)

    # Convert embarked to a number
    df['Embarked'] = df['Embarked'].fillna('N')
    df['NumEmbarked'] = df['Embarked'].map({ 'S': 0, 'C': 1, 'Q': 2, 'N': 3 }).astype(int)

    # Fill nan in Fare
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

    # extract titles (Mr, Mrs, Master) into a new title field
    df['Title'] = df['Name'].str.contains(r'Mr\.').map( { True: 0, False: -1 } )
    df.loc[df['Title'] == -1, 'Title'] = df[df['Title'] == -1]['Name'].str \
                                         .contains(r'Mrs\.').map({True: 1, False: -1})
    df.loc[df['Title'] == -1, 'Title'] = df[df['Title'] == -1]['Name'].str \
                                         .contains(r'Miss\.').map({True: 2, False: -1})
    df.loc[df['Title'] == -1, 'Title'] = df[df['Title'] == -1]['Name'].str \
                                         .contains(r'Master\.').map({True: 3, False: -1})

    if withAges:
        # Drop the the data points where ages are missing if withAges = true
        df = df[df['Age'].notnull()]
    else:
        # if withAges = false, drop the age axis and only return data points without ages
        df = df[df['Age'].isnull()]
        df = df.drop('Age', axis=1)

    # Calculate number of family members on board
    df['FamilyMembers'] = df['SibSp'] + df['Parch']

    # delete unused attributes
    df = df.drop(['Name', 'Ticket', 'Cabin', 'Sex', 'Embarked', 'SibSp', 'Parch'], axis=1)
    print(df.columns.values)

    # df.to_csv(filename+'.processed', index=False)
    # return a tuple of values and passengerIds
    return (df.drop('PassengerId', axis=1).values, df['PassengerId'].values)

def writeOutput(outputArray, passengerIds, filename):
    """ Write the results to an output csv with two columns, PassengerId and Survived """
    # Create a pandas DataFrame
    df = pandas.DataFrame({'PassengerId': passengerIds, 'Survived': outputArray })
    df = df.sort_values(by='PassengerId')
    df.to_csv(filename, index=False)

if __name__ == '__main__':
    # main routine

    # We split the data into two sets, one that contains all the data points where
    # the ages are present, and the other one containing all the data points without
    # ages. We then create two different Random Forests for these sets and use either
    # of those as the estimator, depending on whether or not we know the age of the
    # person whose outcome we want to predict.

    # Use a grid search to find the optimal max depth of the forests
    forest1 = GridSearchCV(RandomForestClassifier(n_estimators = 200),
                          {'max_depth': list(range(1,20)) },
                          cv=5, n_jobs=-1)
    forest2 = GridSearchCV(RandomForestClassifier(n_estimators = 200),
                          {'max_depth': list(range(1,20)) },
                          cv=5, n_jobs=-1)
    trainArray1, passengerIds1 = arrayFromCsv('data/train.csv', withAges=True)
    trainArray2, passengerIds2 = arrayFromCsv('data/train.csv', withAges=False)
    testArray1, passengerIds1 = arrayFromCsv('data/test.csv', withAges=True)
    testArray2, passengerIds2 = arrayFromCsv('data/test.csv', withAges=False)
    trainData1 = trainArray1[0::,1::]
    trainTarget1 = trainArray1[0::,0]
    trainData2 = trainArray2[0::,1::]
    trainTarget2 = trainArray2[0::,0]

    # Fit the training data to the Survived labels and create the decision trees
    forest1 = forest1.fit(trainData1, trainTarget1)
    # run the fitted model on the test data
    output1 = forest1.predict(testArray1).astype(int)

    forest2 = forest2.fit(trainData2, trainTarget2)
    output2 = forest2.predict(testArray2).astype(int)

    print ('Optimal parameters:', forest1.best_params_)
    forest1 = RandomForestClassifier(n_estimators = 200, max_depth=forest1.best_params_['max_depth'])
    scores = cross_validation.cross_val_score(forest1, trainData1, trainTarget1, cv=5)
    print ('Scores: ', scores)
    print ('Mean Score: ', scores.mean())

    output = numpy.concatenate((output1, output2))
    passengerIds = numpy.concatenate((passengerIds1, passengerIds2))
    writeOutput(output, passengerIds, 'data/out.csv')
