from config import *
from common import *

import numpy as np
import pandas as pd

def transform_data(df:pd.DataFrame, model_name:str):
    """Takes dataframe as input and transforms data"""
    # make sure no NAs appear
    df = df.fillna(-3)

    # set the target variable
    df['target_3'] = np.where(df['target_login_days'] <= TARGET_MAX_LOGIN_DAYS, 1, 0)

    # get time dependent variables
    df_rescaled = df[FEATURES]

    # convert categorical variables
    df, cat_dict, cat_inv_dict = cat_transform(df, CAT_TYPE)

    # transform time dependent variables
    no_users = df['unique_identifier'].nunique()

    x = np.swapaxes(df_rescaled[FEATURES].values.reshape(no_users, SEQ_LEN, -1), 1, 2)

    if model_name=='03days':
        days_to_delete = 27
        x = np.delete(x, np.arange(0, days_to_delete), 2)
    elif model_name=='14days':
        days_to_delete = 16
        x = np.delete(x, np.arange(0, days_to_delete), 2)

    # transform time independent variables
    emb_vars = df.groupby(['unique_identifier'])[CAT_TYPE].max().values
    y = df.groupby('unique_identifier')['target_3'].max().values

    # get additional information for later checks
    info_vars = df.groupby(['unique_identifier'])[INFO_TRAIN].max().values

    return x, emb_vars, y, info_vars, cat_dict, cat_inv_dict