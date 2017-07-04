import pandas as pd
import numpy as np
import sys
import os

def load_titanic(file):  
    df = pd.read_csv(file, na_values=['NA'])
    
    y = np.array(df['Survived'])
    df.drop('Survived', axis=1, inplace=True)
    X = df.ix[:,:]
    
    print(X.columns)
    
    uninformative = ['PassengerId', 'Ticket', 'Name', 'Cabin']
    for col in uninformative:
        X.drop(col, axis=1, inplace=True)

    information = [('Pclass', 'category'), ('Name', str), ('Sex', 'category'), 
                   ('Age', np.int8), ('SibSp', 'category'), ('Parch', 'category'), 
                   ('Fare', np.float32), ('Cabin', str), ('Embarked', 'category')
    ]
    X = one_of_k_encoding(['Pclass','Sex', 'SibSp', 'Parch', 'Embarked'], X)
    
    # Check for NaNs and if found impute most common value for now
    from sklearn.preprocessing import Imputer
    impute = Imputer(strategy='median', copy=False)
    X = impute.fit_transform(X)
    
    return X, y

def one_of_k_encoding(cols, df):
    """ Categorial encoding of columns where datatype is categorical. 
        Input:    cols [list]
                  df   [pd.DataFrame]
        Output:   updated df
    """
    assert type(cols)==list, "Expecting a list."
    for col in cols:
        if col in df.columns.values: # Silently skip if not present
            df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
            df.drop(col, axis=1, inplace=True)
    return df



if __name__ == '__main__':
    sys.exit(-1)