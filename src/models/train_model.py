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
max_date_train = '2018-04-11'

min_date_test = '2018-10-16'
max_date_test = '2019-04-10'

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


pred_test = mod1.predict_proba(test1_x)
pred_train = mod1.predict_proba(train1_x)

pred_calib_test = mod_calib.predict_proba(test1_x)
pred_calib_train = mod_calib.predict_proba(train1_x)

test1['pred1'] = pred_test[:,1]
train1['pred1'] = pred_train[:,1]

test1['pred1_calib'] = pred_calib_test[:,1]
train1['pred1_calib'] = pred_calib_train[:,1]

test1['pred_prob_team1'] = test1['pred1']
test1['pred_prob_team2'] = 1 - test1['pred_prob_team1']

test1['pred_prob_calib_team1'] = test1['pred1_calib']
test1['pred_prob_calib_team2'] = 1 - test1['pred_prob_calib_team1']

train1['pred_prob_team1'] = train1['pred1']
train1['pred_prob_team2'] = 1 - train1['pred_prob_team1']

train1['pred_prob_calib_team1'] = train1['pred1_calib']
train1['pred_prob_calib_team2'] = 1 - train1['pred_prob_calib_team1']

test1['pred_win'] = (test1['pred1'] >= 0.5)*1
train1['pred_win'] = (train1['pred1'] >= 0.5)*1

test1['pred_win_calib'] = (test1['pred1_calib'] >= 0.5)*1
train1['pred_win_calib'] = (train1['pred1_calib'] >= 0.5)*1


# auc
auc_test = metrics.roc_auc_score(test1[target],test1['pred1'])
auc_train = metrics.roc_auc_score(train1[target],train1['pred1'])

auc_calib_test = metrics.roc_auc_score(test1[target],test1['pred1_calib'])
auc_calib_train = metrics.roc_auc_score(train1[target],train1['pred1_calib'])


# log loss
logloss_test = metrics.log_loss(test1[target],test1['pred1'])
logloss_train = metrics.log_loss(train1[target],train1['pred1'])

logloss_calib_test = metrics.log_loss(test1[target], test1['pred_prob_calib_team1'])
logloss_calib_train = metrics.log_loss(train1[target], train1['pred_prob_calib_team1'])


accuracy_test = accuracy(test1['pred_win'], test1['WIN_HOME'])
accuracy_train = accuracy(train1['pred_win'], train1['WIN_HOME'])

accuracy_calib_test = accuracy(test1['pred_win_calib'], test1['WIN_HOME'])
accuracy_calib_train = accuracy(train1['pred_win_calib'], train1['WIN_HOME'])

model = mod1
plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plot_importance(mod1)
pyplot.show()

print(mod1.feature_importances_)

mod1.get_booster().get_score().items()
mapper = {'f{0}'.format(i): v for i, v in enumerate(predictors)}
mapped = {mapper[k]: v for k, v in mod1.get_booster().get_score().items()}
mapped

xgb.plot_importance(mapped, color='red', max_num_features=30, importance_type='gain')

imp_plot.tight_layout()
plt.tight_layout()

def plot_xgb_feat_importance(model_obj, predictors, importance_type='gain', color='red', max_num_features=50):
    mapper = {'f{0}'.format(i): v for i, v in enumerate(predictors)}
    mapped = {mapper[k]: v for k, v in model_obj.get_booster().get_score().items()}
    fig = plt.figure(dpi=180)
    ax = plt.subplot(1,1,1)
    xgb.plot_importance(mapped, color=color, max_num_features=max_num_features, importance_type=importance_type)

    plt.tight_layout()
    plt.show()


plot_xgb_feat_importance(mod1, predictors, 'gain', 'blue', 50)
