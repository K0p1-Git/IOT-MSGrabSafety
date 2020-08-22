import pandas as pd
import numpy as np
import pickle

from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing

from sklearn.metrics import classification_report, roc_curve, confusion_matrix, accuracy_score, roc_auc_score, auc

def dispersion(s):
    # calculate range
    return s.max() - s.min()

def feature_eng(data):
    # required column names
    COL_ACCE = ('acceleration_x', 'acceleration_y', 'acceleration_z')
    COL_GYRO = ('gyro_x', 'gyro_y', 'gyro_z')
    COL_TELE = ('bookingID', 'Accuracy', 'Bearing', 'second', 'Speed', 'acceleration_x', 'acceleration_y', 'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z')

    bid = data.bookingID.unique()

    # Data Validation 

    if not sorted(data.columns) == sorted(COL_TELE):
        raise Exception('Input columns mismatched! Expected: \n {}'.format(COL_TELE))

    # sort according to bookingID & seconds
    df_use = data.sort_values(['bookingID', 'second']).reset_index(drop=True)

    # drop 'Accuracy' & 'Bearing' to save memory. we don't need these anymore. 
    df_use.drop(['Accuracy', 'Bearing'], axis=1, inplace=True)

    # transform triaxial gyro readings into its first principal components
    pca_gyro = PCA().fit(df_use.loc[:, ['gyro_x', 'gyro_y', 'gyro_z']])
    pca_gyro.explained_variance_ratio_

    # Data Transformation

    # calculate magnitude of acceleration sqrt(acc_x^2 + acc_y^2 + acc_z^2)

    df_use['acceleration'] = np.sqrt((df_use.loc[:, COL_ACCE] ** 2).sum(axis=1))

    df_use['gyro'] = pca_gyro.transform(df_use.loc[:, COL_GYRO])[:,0]

    # Aggregating the features

    feature1 = df_use.groupby('bookingID')['acceleration', 'gyro', 'Speed', 'second'].agg(['mean', 'median', 'std', dispersion]).fillna(0)
    feature1.columns = ['_'.join(col) for col in feature1.columns] # rename columns
    feature1.reset_index(inplace=True)

    output = pd.DataFrame(bid, columns=['bookingID'])
    output = output.merge(feature1, how='left', on='bookingID')
  
    return output

def predict(res):
    model = pickle.load (open("./models/saved_model.pkl", "rb"))
    
    bid = res.bookingID.unique()
    res1 = res.drop(['bookingID'],axis=1)
    
    result = model.predict(res1)
    print("Prediction returned")
    return(result)