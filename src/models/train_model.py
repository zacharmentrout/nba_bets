#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 2010

@author: zach
"""

import pandas as pd
import matplotlib
import csv
import os
import sklearn as sk
import xgboost as xgb
import math
from pathlib import Path
import numpy as np
from datetime import datetime as dt
from sklearn import metrics   #Additional scklearn functions
import itertools as it
import sys
import random
import multiprocessing as mp
from numpy.random import seed
from functools import reduce

from xgboost.sklearn import XGBClassifier, XGBModel
from xgboost import plot_importance

from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

import matplotlib.pyplot as plt
from pylab import plot, show, subplot, specgram, imshow, savefig
from sklearn.ensemble.partial_dependence import plot_partial_dependence

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt

data_location = '/Users/zach/Documents/git/nba_bets/data/'
in_delim = '|'

data_dir = Path(data_location)
processed_dir = data_dir / 'processed'
raw_dir = data_dir / 'raw'


##################
# functions
##################
def accuracy(pred, obs):
    if len(pred) != len(obs):
        raise ValueError('Predictions and observations different lengths')
    return(sum((pred == obs)*1) / len(pred))

def plot_xgb_feat_importance(model_obj, predictors, importance_type='gain', color='red', max_num_features=50):
    mapper = {'f{0}'.format(i): v for i, v in enumerate(predictors)}
    mapped = {mapper[k]: v for k, v in model_obj.get_booster().get_score().items()}
    fig = plt.figure(dpi=180)
    ax = plt.subplot(1,1,1)
    xgb.plot_importance(mapped, color=color, max_num_features=max_num_features, importance_type=importance_type)

    plt.tight_layout()
    plt.show()

def calc_model_performance(df, col_name_pred='pred', col_name_obs='obs',type='classification'):
    if type == 'classification':
        df[col_name_pred + '_01'] = (df[col_name_pred] >= 0.5) * 1
        out = pd.DataFrame()
        out['auc'] = [metrics.roc_auc_score(df[col_name_obs],df[col_name_pred])]
        out['logloss'] = [metrics.log_loss(df[col_name_obs],df[col_name_pred])]
        out['accuracy'] = [accuracy(df[col_name_pred + '_01'],df[col_name_obs])]
    else:
        raise ValueError('incorrect type')

    return(out)

def assign_home_away_prob(train, test):
    test['pred_prob_HOME'] = test['pred']
    test['pred_prob_AWAY'] = 1 - test['pred_prob_HOME']

    test['pred_prob_calib_HOME'] = test['pred_calib']
    test['pred_prob_calib_AWAY'] = 1 - test['pred_prob_calib_HOME']

    train['pred_prob_HOME'] = train['pred']
    train['pred_prob_AWAY'] = 1 - train['pred_prob_HOME']

    train['pred_prob_calib_HOME'] = train['pred_calib']
    train['pred_prob_calib_AWAY'] = 1 - train['pred_prob_calib_HOME']

    test['pred_win_HOME'] = (test['pred'] >= 0.5)*1
    test['pred_win_AWAY'] = 1 - test['pred_win_HOME']

    train['pred_win_HOME'] = (train['pred'] >= 0.5)*1
    train['pred_win_AWAY'] = (train['pred'] >= 0.5)*1

    test['pred_win_calib_HOME'] = (test['pred_calib'] >= 0.5)*1
    test['pred_win_calib_AWAY'] = 1 - test['pred_win_calib_HOME']

    train['pred_win_calib_HOME'] = (train['pred_calib'] >= 0.5)*1
    train['pred_win_calib_AWAY'] = 1 - train['pred_win_calib_HOME']

    return(train, test)

##################
# read in data
##################
train_data_original = pd.read_csv(processed_dir / 'training_data.csv', in_delim)

# limit to regular season
train_data_reg = train_data_original[train_data_original['SEASON_TYPE'] == 'Regular Season']

train_data_init = train_data_reg[[x1 >= 10 and x2 >= 10 for x1,x2 in zip(train_data_reg['TEAM_FEATURE_cumulative_count_GAME_NUMBER_HOME'], train_data_reg['TEAM_FEATURE_cumulative_count_GAME_NUMBER_AWAY'])]]

train_data_init = train_data_init[train_data_init['GAME_DATE'] <= '2019-08-22']

dim_season = pd.read_csv(raw_dir / 'dim_season.csv', sep='|')
#train_data_init.to_csv(processed_dir / 'train_data_test.csv', sep=',')

# train_data_init = train_data_init.merge(dim_season[['SEASON_NUMBER', 'min_GAME_DATE', 'max_GAME_DATE']], how='left', on='SEASON_NUMBER')

target = 'WIN_HOME'


predictors = [
'TEAM_FEATURE_cumulative_count_GAME_NUMBER_AWAY'
,'TEAM_FEATURE_cumulative_count_GAME_NUMBER_HOME'
,'TEAM_FEATURE_TEAM_IS_HOME_TEAM_cumulative_sum_AWAY'
,'TEAM_FEATURE_TEAM_IS_HOME_TEAM_cumulative_sum_HOME'
,'TEAM_FEATURE_TEAM_IS_AWAY_TEAM_cumulative_sum_AWAY'
,'TEAM_FEATURE_TEAM_IS_AWAY_TEAM_cumulative_sum_HOME'
,'TEAM_FEATURE_MIN_cumulative_sum_AWAY'
,'TEAM_FEATURE_MIN_cumulative_sum_HOME'
,'TEAM_FEATURE_WIN_cumulative_sum_AWAY'
,'TEAM_FEATURE_WIN_cumulative_sum_HOME'
,'TEAM_FEATURE_LOSE_cumulative_sum_AWAY'
,'TEAM_FEATURE_LOSE_cumulative_sum_HOME'
,'TEAM_FEATURE_PTS_cumulative_sum_AWAY'
,'TEAM_FEATURE_PTS_cumulative_sum_HOME'
,'TEAM_FEATURE_PTS_per_min_expanding_mean_AWAY'
,'TEAM_FEATURE_PTS_per_min_expanding_mean_HOME'
,'TEAM_FEATURE_FTM_per_min_expanding_mean_AWAY'
,'TEAM_FEATURE_FTM_per_min_expanding_mean_HOME'
,'TEAM_FEATURE_BLK_per_min_expanding_mean_AWAY'
,'TEAM_FEATURE_BLK_per_min_expanding_mean_HOME'
,'TEAM_FEATURE_REB_per_min_expanding_mean_AWAY'
,'TEAM_FEATURE_REB_per_min_expanding_mean_HOME'
,'TEAM_FEATURE_DREB_per_min_expanding_mean_AWAY'
,'TEAM_FEATURE_DREB_per_min_expanding_mean_HOME'
,'TEAM_FEATURE_FG3A_per_min_expanding_mean_AWAY'
,'TEAM_FEATURE_FG3A_per_min_expanding_mean_HOME'
,'TEAM_FEATURE_FG3M_per_min_expanding_mean_AWAY'
,'TEAM_FEATURE_FG3M_per_min_expanding_mean_HOME'
,'TEAM_FEATURE_FGM_per_min_expanding_mean_AWAY'
,'TEAM_FEATURE_FGM_per_min_expanding_mean_HOME'
,'TEAM_FEATURE_PF_per_min_expanding_mean_AWAY'
,'TEAM_FEATURE_PF_per_min_expanding_mean_HOME'
,'TEAM_FEATURE_AST_per_min_expanding_mean_AWAY'
,'TEAM_FEATURE_AST_per_min_expanding_mean_HOME'
,'TEAM_FEATURE_FGA_per_min_expanding_mean_AWAY'
,'TEAM_FEATURE_FGA_per_min_expanding_mean_HOME'
,'TEAM_FEATURE_STL_per_min_expanding_mean_AWAY'
,'TEAM_FEATURE_STL_per_min_expanding_mean_HOME'
,'TEAM_FEATURE_OREB_per_min_expanding_mean_AWAY'
,'TEAM_FEATURE_OREB_per_min_expanding_mean_HOME'
,'TEAM_FEATURE_PTS_per_min_ewma_AWAY'
,'TEAM_FEATURE_PTS_per_min_ewma_HOME'
,'TEAM_FEATURE_FTM_per_min_ewma_AWAY'
,'TEAM_FEATURE_FTM_per_min_ewma_HOME'
,'TEAM_FEATURE_BLK_per_min_ewma_AWAY'
,'TEAM_FEATURE_BLK_per_min_ewma_HOME'
,'TEAM_FEATURE_REB_per_min_ewma_AWAY'
,'TEAM_FEATURE_REB_per_min_ewma_HOME'
,'TEAM_FEATURE_DREB_per_min_ewma_AWAY'
,'TEAM_FEATURE_DREB_per_min_ewma_HOME'
,'TEAM_FEATURE_FG3A_per_min_ewma_AWAY'
,'TEAM_FEATURE_FG3A_per_min_ewma_HOME'
,'TEAM_FEATURE_FG3M_per_min_ewma_AWAY'
,'TEAM_FEATURE_FG3M_per_min_ewma_HOME'
,'TEAM_FEATURE_FGM_per_min_ewma_AWAY'
,'TEAM_FEATURE_FGM_per_min_ewma_HOME'
,'TEAM_FEATURE_PF_per_min_ewma_AWAY'
,'TEAM_FEATURE_PF_per_min_ewma_HOME'
,'TEAM_FEATURE_AST_per_min_ewma_AWAY'
,'TEAM_FEATURE_AST_per_min_ewma_HOME'
,'TEAM_FEATURE_FGA_per_min_ewma_AWAY'
,'TEAM_FEATURE_FGA_per_min_ewma_HOME'
,'TEAM_FEATURE_STL_per_min_ewma_AWAY'
,'TEAM_FEATURE_STL_per_min_ewma_HOME'
,'TEAM_FEATURE_OREB_per_min_ewma_AWAY'
,'TEAM_FEATURE_OREB_per_min_ewma_HOME'
,'TEAM_FEATURE_TEAM_IS_HOME_TEAM_cumulative_sum_COURT_AWAY'
,'TEAM_FEATURE_TEAM_IS_HOME_TEAM_cumulative_sum_COURT_HOME'
,'TEAM_FEATURE_TEAM_IS_AWAY_TEAM_cumulative_sum_COURT_AWAY'
,'TEAM_FEATURE_TEAM_IS_AWAY_TEAM_cumulative_sum_COURT_HOME'
,'TEAM_FEATURE_MIN_cumulative_sum_COURT_AWAY'
,'TEAM_FEATURE_MIN_cumulative_sum_COURT_HOME'
,'TEAM_FEATURE_WIN_cumulative_sum_COURT_AWAY'
,'TEAM_FEATURE_WIN_cumulative_sum_COURT_HOME'
,'TEAM_FEATURE_LOSE_cumulative_sum_COURT_AWAY'
,'TEAM_FEATURE_LOSE_cumulative_sum_COURT_HOME'
,'TEAM_FEATURE_PTS_cumulative_sum_COURT_AWAY'
,'TEAM_FEATURE_PTS_cumulative_sum_COURT_HOME'
,'TEAM_FEATURE_PTS_per_min_expanding_mean_COURT_AWAY'
,'TEAM_FEATURE_PTS_per_min_expanding_mean_COURT_HOME'
,'TEAM_FEATURE_FTM_per_min_expanding_mean_COURT_AWAY'
,'TEAM_FEATURE_FTM_per_min_expanding_mean_COURT_HOME'
,'TEAM_FEATURE_BLK_per_min_expanding_mean_COURT_AWAY'
,'TEAM_FEATURE_BLK_per_min_expanding_mean_COURT_HOME'
,'TEAM_FEATURE_REB_per_min_expanding_mean_COURT_AWAY'
,'TEAM_FEATURE_REB_per_min_expanding_mean_COURT_HOME'
,'TEAM_FEATURE_DREB_per_min_expanding_mean_COURT_AWAY'
,'TEAM_FEATURE_DREB_per_min_expanding_mean_COURT_HOME'
,'TEAM_FEATURE_FG3A_per_min_expanding_mean_COURT_AWAY'
,'TEAM_FEATURE_FG3A_per_min_expanding_mean_COURT_HOME'
,'TEAM_FEATURE_FG3M_per_min_expanding_mean_COURT_AWAY'
,'TEAM_FEATURE_FG3M_per_min_expanding_mean_COURT_HOME'
,'TEAM_FEATURE_FGM_per_min_expanding_mean_COURT_AWAY'
,'TEAM_FEATURE_FGM_per_min_expanding_mean_COURT_HOME'
,'TEAM_FEATURE_PF_per_min_expanding_mean_COURT_AWAY'
,'TEAM_FEATURE_PF_per_min_expanding_mean_COURT_HOME'
,'TEAM_FEATURE_AST_per_min_expanding_mean_COURT_AWAY'
,'TEAM_FEATURE_AST_per_min_expanding_mean_COURT_HOME'
,'TEAM_FEATURE_FGA_per_min_expanding_mean_COURT_AWAY'
,'TEAM_FEATURE_FGA_per_min_expanding_mean_COURT_HOME'
,'TEAM_FEATURE_STL_per_min_expanding_mean_COURT_AWAY'
,'TEAM_FEATURE_STL_per_min_expanding_mean_COURT_HOME'
,'TEAM_FEATURE_OREB_per_min_expanding_mean_COURT_AWAY'
,'TEAM_FEATURE_OREB_per_min_expanding_mean_COURT_HOME'
,'TEAM_FEATURE_PTS_per_min_ewma_COURT_AWAY'
,'TEAM_FEATURE_PTS_per_min_ewma_COURT_HOME'
,'TEAM_FEATURE_FTM_per_min_ewma_COURT_AWAY'
,'TEAM_FEATURE_FTM_per_min_ewma_COURT_HOME'
,'TEAM_FEATURE_BLK_per_min_ewma_COURT_AWAY'
,'TEAM_FEATURE_BLK_per_min_ewma_COURT_HOME'
,'TEAM_FEATURE_REB_per_min_ewma_COURT_AWAY'
,'TEAM_FEATURE_REB_per_min_ewma_COURT_HOME'
,'TEAM_FEATURE_DREB_per_min_ewma_COURT_AWAY'
,'TEAM_FEATURE_DREB_per_min_ewma_COURT_HOME'
,'TEAM_FEATURE_FG3A_per_min_ewma_COURT_AWAY'
,'TEAM_FEATURE_FG3A_per_min_ewma_COURT_HOME'
,'TEAM_FEATURE_FG3M_per_min_ewma_COURT_AWAY'
,'TEAM_FEATURE_FG3M_per_min_ewma_COURT_HOME'
,'TEAM_FEATURE_FGM_per_min_ewma_COURT_AWAY'
,'TEAM_FEATURE_FGM_per_min_ewma_COURT_HOME'
,'TEAM_FEATURE_PF_per_min_ewma_COURT_AWAY'
,'TEAM_FEATURE_PF_per_min_ewma_COURT_HOME'
,'TEAM_FEATURE_AST_per_min_ewma_COURT_AWAY'
,'TEAM_FEATURE_AST_per_min_ewma_COURT_HOME'
,'TEAM_FEATURE_FGA_per_min_ewma_COURT_AWAY'
,'TEAM_FEATURE_FGA_per_min_ewma_COURT_HOME'
,'TEAM_FEATURE_STL_per_min_ewma_COURT_AWAY'
,'TEAM_FEATURE_STL_per_min_ewma_COURT_HOME'
,'TEAM_FEATURE_OREB_per_min_ewma_COURT_AWAY'
,'TEAM_FEATURE_OREB_per_min_ewma_COURT_HOME'
]


model_param_dict = {
    'learning_rate':[0.05],
    'n_estimators':[25],
    'max_depth':[5],
    'min_child_weight':[1],
    'gamma':[0.5],
    'subsample':[0.8],
    'colsample_bytree':[1.0],
    'objective':['binary:logistic'],
    'nthread':[1],
    'scale_pos_weight':[1],
    'seed':[29]
    ,'train_frac':[0.8]
}



##################
# test model training
##################

# prep
min_date_train = '2007-11-21'
max_date_train = '2017-04-12'

min_date_test = '2017-10-17'
max_date_test = '2018-04-11'

train_frac = 0.8

train_dat = train_data_init[(train_data_init['GAME_DATE'] >= min_date_train) & (train_data_init['GAME_DATE'] <= max_date_train)]

test_dat = train_data_init[(train_data_init['GAME_DATE'] >= min_date_test) & (train_data_init['GAME_DATE'] <= max_date_test)]


model_param_names = sorted(model_param_dict)
model_param_combinations = it.product(*(model_param_dict[Name] for Name in model_param_names))
model_param_list = list(model_param_combinations)
model_param_df = pd.DataFrame(model_param_list, columns=model_param_names)

model_param_df_row = model_param_df.iloc[[0],:]
model_params = {key:val for key, val in zip(model_param_df_row.columns.values, model_param_df_row.iloc[0])}


# train model
np.random.seed(0)
sample_ind = np.random.choice(train_dat.shape[0], size=int(np.floor(train_frac*train_dat.shape[0])),replace=False)

train1 = train_dat.iloc[sample_ind,:].copy()

calib_ind = list(set(range(train_dat.shape[0])) - set(sample_ind))
calib1 = train_dat.iloc[calib_ind,:].copy()

# train
train1_x = train1[predictors]
train1_x.columns = ['f'+str(i) for i in range(len(predictors))]

calib1_x = calib1[predictors]
calib1_x.columns = ['f'+str(i) for i in range(len(predictors))]

test1_x = test_dat[predictors]
test1_x.columns = ['f'+str(i) for i in range(len(predictors))]

#random.seed(seed)
mod1 = XGBClassifier(**model_params)
mod1.fit(train1_x, train1[target])

mod_calib = CalibratedClassifierCV(mod1, method='sigmoid',cv='prefit')
mod_calib.fit(calib1_x, calib1[target])

test_dat['pred'] = mod1.predict_proba(test1_x)[:,1]
train1['pred'] = mod1.predict_proba(train1_x)[:,1]

test_dat['pred_calib'] = mod_calib.predict_proba(test1_x)[:,1]
train1['pred_calib'] = mod_calib.predict_proba(train1_x)[:,1]


train1, test_dat = assign_home_away_prob(train1, test_dat)

perf_test = calc_model_performance(test_dat, 'pred', target)
perf_calib_test = calc_model_performance(test_dat, 'pred_calib', target)
perf_train = calc_model_performance(train1, 'pred', target)
perf_calib_train = calc_model_performance(train1, 'pred_calib', target)

perf_all = pd.concat([perf_test, perf_calib_test, perf_train, perf_calib_train])
perf_all['description'] = ['test', 'test_calib', 'train', 'train_calib']

plot_xgb_feat_importance(mod1, predictors, 'gain', 'blue', 50)
