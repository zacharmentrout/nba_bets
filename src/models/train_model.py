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

import pickle

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

def assign_home_away_prob(df, col_name_pred='pred'):

    # predict probability of win
    df['pred_prob_HOME'] = df[col_name_pred]
    df['pred_prob_AWAY'] = 1 - df['pred_prob_HOME']

    # predict win/lose
    df['pred_win_HOME'] = (df[col_name_pred] >= 0.5)*1
    df['pred_win_AWAY'] = 1 - df['pred_win_HOME']

    return(df)

def assign_home_away_prob_with_xgb_classifier(model_obj, dat, predictors):
    dat['pred'] = predict_with_xgb_classifier(model_obj, dat, predictors)
    out_df = pd.DataFrame(zip(dat['pred'], 1 - dat['pred']), columns=['pred_prob_HOME', 'pred_prob_AWAY'])

    return(out_df)



def predict_with_xgb_classifier(model_obj, dat, predictors=None):
    dat_x = dat[predictors]
    dat_x.columns = ['f'+str(i) for i in range(len(predictors))]
    pred = (model_obj.predict_proba(dat_x))[:,1]
    return(pred)

def train_xgb_classifier(dat, predictors, target_col, params):

    if params['train_frac'] < 1.0:
        sample_ind = np.random.choice(dat.shape[0], size=int(np.floor(params['train_frac']*dat.shape[0])),replace=False)
        train_dat = dat.iloc[sample_ind,:].copy()
        calib_ind = list(set(range(dat.shape[0])) - set(sample_ind))
        calib_dat = dat.iloc[calib_ind,:].copy()
        calib_dat_x = calib_dat[predictors]
        calib_dat_x.columns = ['f'+str(i) for i in range(len(predictors))]
    else:
        train_dat = dat.copy()

    train_dat_x = train_dat[predictors]
    train_dat_x.columns = ['f'+str(i) for i in range(len(predictors))]

    mod = XGBClassifier(**params)
    mod.fit(train_dat_x, train_dat[target_col])

    if params['train_frac'] < 1.0:
        mod = CalibratedClassifierCV(mod, method='sigmoid',cv='prefit')
        mod.fit(calib_dat_x, calib_dat[target])

    return(mod)

def train_test_model(train_dat, test_dat, train_fn, pred_fn, predictors, calc_performance=True, return_data=True, return_model_object=True):

    mod = train_fn(dat=train_dat, predictors=predictors, target_col=target, params=model_params)

    test_dat['pred'] = pred_fn(model_obj=mod, dat=test_dat, predictors=predictors)

    train_dat['pred'] = pred_fn(model_obj=mod, dat=train_dat, predictors=predictors)

    test_dat = assign_home_away_prob(test_dat)
    train_dat = assign_home_away_prob(train_dat)

    return_dict = {
        'performance':None,
        'train_data':None,
        'test_data':None,
        'model_object':None
    }

    if calc_performance:
        perf_test = calc_model_performance(test_dat, 'pred', target)
        perf_train = calc_model_performance(train_dat, 'pred', target)

        perf_all = pd.concat([perf_test, perf_train])
        perf_all['dataset'] = ['test', 'train']
        perf_all['model_type'] = type(mod).__name__

        return_dict['performance'] = perf_all

    if return_data:
        return_dict['train_data'] = train_dat
        return_dict['test_data'] = test_dat

    if return_model_object:
        return_dict['model_object'] = mod

    return(return_dict)


class Modeler:
    def __init__(self, parameters, training_data, training_function, prediction_function, performance_function, predictors, target, prediction_col='pred', prob_home_col='prob_WIN_HOME', prob_away_col='prob_WIN_AWAY'):
        self.parameters = parameters
        self.training_data = training_data
        self.training_function = training_function
        self.prediction_function = prediction_function
        self.performance_function = performance_function
        self.predictors = predictors
        self.target = target
        self.model_object = None
        self.prediction_col = prediction_col
        self.prob_home_col = prob_home_col
        self.prob_away_col = prob_away_col

    def train_model(self):
        self.model_object = self.training_function(self.training_data, self.predictors, self.target, self.parameters)

    def predict(self, test_data):
        preds = self.prediction_function(self.model_object, test_data, self.predictors)
        return(preds)

    def assign_home_away_probabilities(self, dat):
        dat[self.prob_home_col] = dat[self.prediction_col]
        dat[self.prob_away_col] = 1 - dat[self.prob_home_col]
        return(dat)

class Strategy:
    def __init__(self, parameters, bet_recommendation_function, performance_function, odds_home_col='odds_dec_home', odds_away_col='odds_dec_away', bet_home_col='place_bet_HOME', bet_away_col='place_bet_AWAY'):
        self.parameters = parameters
        self.bet_recommendation_function = bet_recommendation_function
        self.performance_function = performance_function
        self.odds_home_col = odds_home_col
        self.odds_away_col = odds_away_col
        self.bet_home_col = bet_home_col
        self.bet_away_col = bet_away_col

    def recommend_bets(self, test_data, **kwargs):
        return(self.bet_recommendation_function(test_data, **kwargs))

    def calc_performance(self, test_data, **kwargs):
        return(self.performance_function(test_data, **kwargs))

# TODO add modeler/strategy init
class Recommender:
    def __init__(self, parameters, training_data, training_function,  prediction_function, model_performance_function, predictors, target, bet_recommendation_function, bet_performance_function, prediction_col='pred', prob_home_col='prob_WIN_HOME', prob_away_col='prob_WIN_AWAY', odds_home_col='odds_dec_home', odds_away_col='odds_dec_away', bet_home_col='place_bet_HOME', bet_away_col='place_bet_AWAY'):

        self.parameters = parameters
        self.modeler = Modeler(parameters, training_data, training_function, prediction_function, model_performance_function, predictors, target, prob_home_col, prob_away_col)
        self.strategy = Strategy(parameters, bet_recommendation_function, bet_performance_function, odds_home_col, odds_away_col, bet_home_col, bet_away_col)
        self.prediction_col = prediction_col
        self.prob_home_col = prob_home_col
        self.prob_away_col = prob_away_col
        self.odds_home_col = odds_home_col
        self.odds_away_col = odds_away_col
        self.bet_home_col = bet_home_col
        self.bet_away_col = bet_away_col

    def train_model(self):
        self.modeler.train_model()

    def predict_and_assign_home_away_probabilities(self, test_data):
        test_data[self.prediction_col] = self.modeler.predict(test_data)
        test_data = self.modeler.assign_home_away_probabilities(test_data)
        return(test_data)

    def recommend_bets(self, test_data, **kwargs):
        return(self.strategy.recommend_bets(test_data, **kwargs))

    def calc_performance(self, test_data, type='both', **kwargs):
        perf = {}
        if type in set(['both', 'bet']):
            perf['bet'] = strategy.calc_performance(test_data, **kwargs)
        elif type in set(['both', 'model']):
            perf['model'] = modeler.calc_performance(test_data, **kwargs)
        else:
            raise ValueError('Argument "type" must be one of ["bet", "model", "both"]')
        return(perf)

##################
# read in data
##################

data_location = '/Users/zach/Documents/git/nba_bets/data/'
in_delim = '|'

target = 'WIN_HOME'

predictors = [
'TEAM_FEATURE_cumulative_win_pct_COURT_HOME'
,'TEAM_FEATURE_cumulative_win_pct_COURT_AWAY'
,'TEAM_FEATURE_cumulative_win_pct_HOME'
,'TEAM_FEATURE_cumulative_win_pct_AWAY'
,'TEAM_FEATURE_cumulative_count_GAME_NUMBER_AWAY'
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
,'TEAM_FEATURE_OREB_per_min_ewma_COURT_HOME',
'TEAM_FEATURE_PTS_TOTAL_cumulative_sum_AWAY', 'TEAM_FEATURE_PTS_TOTAL_per_min_ewma_AWAY', 'TEAM_FEATURE_PTS_TOTAL_per_min_expanding_mean_COURT_AWAY', 'TEAM_FEATURE_cumulative_pt_pct_COURT_HOME', 'TEAM_FEATURE_PTS_TOTAL_cumulative_sum_HOME', 'TEAM_FEATURE_PTS_TOTAL_per_min_ewma_COURT_HOME', 'TEAM_FEATURE_PTS_TOTAL_per_min_expanding_mean_AWAY', 'TEAM_FEATURE_PTS_TOTAL_per_min_ewma_COURT_AWAY', 'TEAM_FEATURE_PTS_TOTAL_cumulative_sum_COURT_HOME', 'TEAM_FEATURE_PTS_TOTAL_per_min_expanding_mean_COURT_HOME', 'TEAM_FEATURE_PTS_TOTAL_per_min_ewma_HOME', 'TEAM_FEATURE_PTS_TOTAL_per_min_expanding_mean_HOME', 'TEAM_FEATURE_cumulative_pt_pct_COURT_AWAY', 'TEAM_FEATURE_PTS_TOTAL_cumulative_sum_COURT_AWAY'
]


model_param_dict = {
    'learning_rate':[0.05],
    'n_estimators':[5],
    'max_depth':[5, 7],
    'min_child_weight':[1],
    'gamma':[0.5],
    'subsample':[0.8],
    'colsample_bytree':[1.0],
    'objective':['binary:logistic'],
    'nthread':[1],
    'scale_pos_weight':[1],
    'seed':[29],
    'train_frac':[0.8],
}


model_param_row_number = 0

##################
# get model parameters
##################
model_param_names = sorted(model_param_dict)
model_param_combinations = it.product(*(model_param_dict[Name] for Name in model_param_names))
model_param_list = list(model_param_combinations)
model_param_df = pd.DataFrame(model_param_list, columns=model_param_names)

model_param_df_row = model_param_df.iloc[[model_param_row_number],:]
model_params = {key:val for key, val in zip(model_param_df_row.columns.values, model_param_df_row.iloc[0])}


##################
# read in data
##################

data_dir = Path(data_location)
processed_dir = data_dir / 'processed'
raw_dir = data_dir / 'raw'

train_data_original = pd.read_csv(processed_dir / 'training_data.csv', in_delim)

# limit to regular season
train_data_reg = train_data_original[train_data_original['SEASON_TYPE'] == 'Regular Season']

train_data_init = train_data_reg[[x1 >= 10 and x2 >= 10 for x1,x2 in zip(train_data_reg['TEAM_FEATURE_cumulative_count_GAME_NUMBER_HOME'], train_data_reg['TEAM_FEATURE_cumulative_count_GAME_NUMBER_AWAY'])]]

train_data_init = train_data_init[train_data_init['GAME_DATE'] <= '2019-08-22']

dim_season = pd.read_csv(raw_dir / 'dim_season.csv', sep='|')

##################
# test model training
##################

# prep
min_date_train = '2007-11-21'
max_date_train = '2016-04-13'

min_date_test = '2016-10-25'
max_date_test = '2017-04-12'

train_dat = train_data_init[(train_data_init['GAME_DATE'] >= min_date_train) & (train_data_init['GAME_DATE'] <= max_date_train)]

test_dat = train_data_init[(train_data_init['GAME_DATE'] >= min_date_test) & (train_data_init['GAME_DATE'] <= max_date_test)]

calc_performance = True
return_data = True
return_model_object = True

train_fn = train_xgb_classifier
pred_fn = predict_with_xgb_classifier

modeler = Modeler(parameters=model_params, training_data=train_dat, training_function=train_xgb_classifier, prediction_function=predict_with_xgb_classifier, performance_function=calc_model_performance, predictors=predictors, target=target)

modeler.train_model()

recommender = Recommender( parameters=model_params, training_data=train_dat, training_function=train_xgb_classifier,  prediction_function=predict_with_xgb_classifier, model_performance_function=calc_model_performance, predictors=predictors, target=target, bet_recommendation_function=None, bet_performance_function=None)


recommender.train_model()

test_dat = modeler.assign_home_away_probabilities(test_dat)

test_dat = recommender.predict_and_assign_home_away_probabilities(test_dat)



results = train_test_model(train_dat=train_dat, test_dat=test_dat, train_fn=train_xgb_classifier, pred_fn=assign_home_away_prob_with_xgb_classifier, predictors=predictors)


print(results['performance'])
mod = results['model_object']

plot_xgb_feat_importance(mod.base_estimator, predictors, 'gain', 'blue', 30)
