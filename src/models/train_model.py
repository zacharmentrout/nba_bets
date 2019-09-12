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
import scipy.optimize as sco


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

import quadprog


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

def calc_model_performance(df, col_name_pred='pred', col_name_obs='WIN_HOME', type='classification', **kwargs):
    if type == 'classification':
        df[col_name_pred + '_01'] = (df[col_name_pred] >= 0.5) * 1
        out = pd.DataFrame()
        if df.shape[0] > 5:
            out['auc'] = [metrics.roc_auc_score(df[col_name_obs],df[col_name_pred])]
            out['logloss'] = [metrics.log_loss(df[col_name_obs],df[col_name_pred])]

        else:
            out['auc'] = -1
            out['logloss'] = -1

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
        if params['seed'] is not None:
            np.random.seed(params['seed'])
        else:
            np.random.seed(123)
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

def rec_bets_simple(test_data, **kwargs):
    if 'bet_budget' in kwargs.keys():
        bet_budget = kwargs['bet_budget']
    else:
        bet_budget = 1

    n_games = test_data.shape[0]
    bet_amount = bet_budget / n_games

    test_data['place_bet_home'] = (test_data['prob_WIN_HOME'] >= 0.5)*1
    test_data['place_bet_away'] = (test_data['prob_WIN_AWAY'] >= 0.5)*1

    test_data['bet_home'] = (test_data['prob_WIN_HOME'] >= 0.5)*1*bet_amount
    test_data['bet_away'] = (test_data['prob_WIN_AWAY'] >= 0.5)*1*bet_amount
    return(test_data)

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

def calc_total_profit(test_data, **kwargs):
    out = pd.DataFrame()
    test_data['profit_home'] = test_data['bet_home'] * (test_data['odds_dec_home'] * test_data['place_bet_home'] * test_data['WIN_HOME'] - 1)
    test_data['profit_away'] = test_data['bet_away'] * (test_data['odds_dec_away'] * test_data['place_bet_away'] * (1-test_data['WIN_HOME']) - 1)
    test_data['profit'] = test_data['profit_home'] + test_data['profit_away']

    test_data['correct_bet_home'] = test_data['place_bet_home'] * test_data['WIN_HOME']
    test_data['correct_bet_away'] = test_data['place_bet_away'] * (1 - test_data['WIN_HOME'])
    test_data['correct_bet'] = test_data['correct_bet_home'] + test_data['correct_bet_away']

    out['total_profit'] = [test_data['profit'].sum()]
    out['total_exp_profit'] = [test_data['exp_profit'].sum()]
    out['total_bets_placed'] = [test_data['place_bet'].sum()]
    out['total_correct_bets'] = [test_data['correct_bet'].sum()]
    out['total_games'] = [test_data.shape[0]]
    out['total_available_bets'] = [test_data.shape[0] - test_data['odds_home'].isna().sum()]


    return(out)


def indicate(x):
    res = [1 if z > 0 else 0 for z in x]
    return(res)


def calc_bet_distribution(test_data, **kwargs):
    test_data['positive_expectation_HOME'] = indicate(test_data['prob_WIN_HOME'] - 1/test_data['odds_dec_home'])
    test_data['positive_expectation_AWAY'] = indicate((1 - test_data['prob_WIN_HOME']) - 1/test_data['odds_dec_away'])

    if 'bet_dist_type' in kwargs.keys():
        bet_dist_type = kwargs['bet_dist_type']
    else:
        bet_dist_type = 'unif'

    if 'bet_budget' in kwargs.keys():
        bet_budget = kwargs['bet_budget']
    else:
        bet_budget = 1

    if bet_dist_type == 'opt':
        max_sharpe_ratio(test_data)
        test_data['bet_home'] = test_data['bet_budget_pct_home'] * bet_budget
        test_data['bet_away'] = test_data['bet_budget_pct_away'] * bet_budget
        test_data['place_bet_home'] = (test_data['bet_home'] > 0.001)*1
        test_data['place_bet_away'] = (test_data['bet_away'] > 0.001)*1
        test_data['place_bet'] = test_data['place_bet_home'] + test_data['place_bet_away']
        return(test_data)


    if bet_dist_type == 'max-ep':
        test_data['calc_exp_profit_home'] = test_data['prob_WIN_HOME'] * test_data['odds_home'] - 1
        test_data['calc_exp_profit_away'] = (1 - test_data['prob_WIN_HOME']) * test_data['odds_away'] - 1
        max_ep_home = np.max(test_data['calc_exp_profit_home'])
        max_ep_away = np.max(test_data['calc_exp_profit_away'])

        test_data['bet_budget_pct_home'] = 0
        test_data['bet_budget_pct_away'] = 0

        if max_ep_home >= max_ep_away:
            max_ep_ind = list(test_data['calc_exp_profit_home']).index(max_ep_home)
            test_data.loc[max_ep_ind,'bet_budget_pct_home'] = 1
        else:
            max_ep_ind = list(test_data['calc_exp_profit_away']).index(max_ep_away)
            test_data.loc[max_ep_ind,'bet_budget_pct_away'] = 1
        test_data['bet_home'] = test_data['bet_budget_pct_home'] * bet_budget
        test_data['bet_away'] = test_data['bet_budget_pct_away'] * bet_budget
        test_data['place_bet_home'] = (test_data['bet_home'] > 0.001)*1
        test_data['place_bet_away'] = (test_data['bet_away'] > 0.001)*1
        test_data['place_bet'] = test_data['place_bet_home'] + test_data['place_bet_away']
        return(test_data)


    if bet_dist_type == 'unif':
        test_data['Bi_home'] = 1
        test_data['Bi_away'] = 1
    elif bet_dist_type == 'conf':
        test_data['Bi_home'] = test_data['prob_WIN_HOME']
        test_data['Bi_away'] = 1 - test_data['prob_WIN_HOME']
    elif bet_dist_type == 'abs-disc':
        test_data['Bi_home'] = test_data['positive_expectation_HOME']
        test_data['Bi_away'] = test_data['positive_expectation_AWAY']
    elif bet_dist_type == 'rel-disc':
        test_data['Bi_home'] = test_data['positive_expectation_HOME'] / test_data['prob_WIN_HOME']
        test_data['Bi_away'] = test_data['positive_expectation_AWAY'] / (1 - test_data['prob_WIN_HOME'])
    else:
        raise ValueError('Invalid bet_dist_type')

    test_data['numer_i_home'] = test_data['Bi_home'] * test_data['positive_expectation_HOME']
    test_data['numer_i_away'] = test_data['Bi_away'] * test_data['positive_expectation_AWAY']
    denom = (test_data['numer_i_home'] + test_data['numer_i_away']).sum()
    test_data['bet_budget_pct_home'] = test_data['numer_i_home'] / denom
    test_data['bet_budget_pct_away'] = test_data['numer_i_away'] / denom
    test_data['bet_home'] = test_data['bet_budget_pct_home'] * bet_budget
    test_data['bet_away'] = test_data['bet_budget_pct_away'] * bet_budget
    test_data['place_bet_home'] = (test_data['bet_home'] > 0.001)*1
    test_data['place_bet_away'] = (test_data['bet_away'] > 0.001)*1
    test_data['place_bet'] = test_data['place_bet_home'] + test_data['place_bet_away']

    return(test_data)


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

    def calc_performance(self, test_data):
        return(self.performance_function(test_data, **self.parameters))

class Strategy:
    def __init__(self, parameters, bet_recommendation_function, performance_function, odds_home_col='odds_dec_home', odds_away_col='odds_dec_away', bet_home_col='place_bet_HOME', bet_away_col='place_bet_AWAY'):
        self.parameters = parameters
        self.bet_recommendation_function = bet_recommendation_function
        self.performance_function = performance_function
        self.odds_home_col = odds_home_col
        self.odds_away_col = odds_away_col
        self.bet_home_col = bet_home_col
        self.bet_away_col = bet_away_col

    def recommend_bets(self, test_data):
        return(self.bet_recommendation_function(test_data, **self.parameters))

    def calc_performance(self, test_data):
        return(self.performance_function(test_data, **self.parameters))

# TODO add modeler/strategy init
class Recommender:
    def __init__(self, parameters, training_data, training_function,  prediction_function, model_performance_function, predictors, target, bet_recommendation_function, bet_performance_function, prediction_col='pred', prob_home_col='prob_WIN_HOME', prob_away_col='prob_WIN_AWAY', odds_home_col='odds_dec_home', odds_away_col='odds_dec_away', bet_home_col='bet_home', bet_away_col='bet_away'):

        self.parameters = parameters
        self.modeler = Modeler(parameters, training_data, training_function, prediction_function, model_performance_function, predictors, target, prediction_col, prob_home_col, prob_away_col)
        self.strategy = Strategy(parameters, bet_recommendation_function, bet_performance_function, odds_home_col, odds_away_col, bet_home_col, bet_away_col)
        self.prediction_col = prediction_col
        self.prob_home_col = prob_home_col
        self.prob_away_col = prob_away_col
        self.odds_home_col = odds_home_col
        self.odds_away_col = odds_away_col
        self.bet_home_col = bet_home_col
        self.bet_away_col = bet_away_col
        self.target = target

    def train_model(self):
        self.modeler.train_model()

    def predict_and_assign_home_away_probabilities(self, test_data):
        test_data[self.prediction_col] = self.modeler.predict(test_data)
        self.modeler.assign_home_away_probabilities(test_data)
        return(test_data)

    def recommend_bets(self, test_data):
        return(self.strategy.recommend_bets(test_data))

    def calc_performance(self, test_data, type='both'):
        if type not in set(['both', 'bet', 'model']):
            raise ValueError('type argument must be one of {both, bet, model}')
        perf = {}
        if type in set(['both', 'bet']):
            perf['bet'] = self.strategy.calc_performance(test_data)
        if type in set(['both', 'model']):
            perf['model'] = self.modeler.calc_performance(test_data)
        return(perf)


def calc_expected_profit(test_data):
    test_data['exp_profit_home'] = (test_data['prob_WIN_HOME'] * test_data['odds_dec_home'] - 1) * test_data['bet_home']
    test_data['exp_profit_away'] = (test_data['prob_WIN_AWAY'] * test_data['odds_dec_away'] - 1) * test_data['bet_away']
    test_data['exp_profit'] = test_data['exp_profit_home'] + test_data['exp_profit_away']
    return(test_data)

def expected_cumulative_profit2(b, p, o):
    result = sum((p * o - 1) * b)
    return(result)

def expected_cumulative_profit(test_data):
    test_data = calc_expected_profit(test_data)
    return(test_data['exp_profit'].sum())

def variance_cumulative_profit(test_data):
    sum1 = ((1 - test_data['prob_WIN_HOME']) * test_data['prob_WIN_HOME'] * pow(test_data['bet_home'], 2) * pow(test_data['odds_dec_home'], 2)).sum()
    sum2 = ((1 - test_data['prob_WIN_AWAY']) * test_data['prob_WIN_AWAY'] * pow(test_data['bet_away'], 2) * pow(test_data['odds_dec_away'], 2)).sum()
    return(sum1+sum2)

def variance_cumulative_profit2(b, p, o):
    result = sum((1 - p) * p * pow(b, 2) * pow(o, 2))
    return(result)

def sd_cumulative_profit2(b, p, o):
    return(sqrt(variance_cumulative_profit2(b,p,o)))

def sharpe_ratio(b,p,o):
    result = expected_cumulative_profit(test_data) / sqrt(variance_cumulative_profit(test_data))
    return(result)

def sharpe_ratio2(b, p, o):
    result = expected_cumulative_profit2(b, p, o) / sd_cumulative_profit2(b, p, o)
    return(result)

def negative_sharpe_ratio(test_data):
    return -1 * sharpe_ratio(test_data)


def negative_sharpe_ratio2(b, p, o):
    return -1 * sharpe_ratio2(b, p, o)

def max_sharpe_ratio2(p, o):
    if len(p) != len(o):
        raise ValueError('Arguments not the same length')
    num_assets = len(p)
    args = (p, o)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(negative_sharpe_ratio2, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    best_bets = result['x']
    return best_bets

def max_sharpe_ratio(test_data):
    dat_copy = test_data.copy()
    valid_rows = dat_copy.index[[False if x else True for x in dat_copy['odds_dec_home'].isna()]].tolist()

    dat_copy = dat_copy.loc[valid_rows]

    p = dat_copy['prob_WIN_HOME'].append(dat_copy['prob_WIN_AWAY'])
    o = dat_copy['odds_dec_home'].append(dat_copy['odds_dec_away'])
    b = max_sharpe_ratio2(p, o)
    b_home = b[range(int(len(b)/2))]
    b_away = b[range(int(len(b)/2), len(b))]
    test_data.loc[valid_rows, 'bet_budget_pct_home'] = b_home
    test_data.loc[valid_rows, 'bet_budget_pct_away'] = b_away
    return(test_data)



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
'TEAM_FEATURE_PTS_TOTAL_cumulative_sum_AWAY', 'TEAM_FEATURE_PTS_TOTAL_per_min_ewma_AWAY', 'TEAM_FEATURE_PTS_TOTAL_per_min_expanding_mean_COURT_AWAY', 'TEAM_FEATURE_cumulative_pt_pct_COURT_HOME', 'TEAM_FEATURE_PTS_TOTAL_cumulative_sum_HOME', 'TEAM_FEATURE_PTS_TOTAL_per_min_ewma_COURT_HOME', 'TEAM_FEATURE_PTS_TOTAL_per_min_expanding_mean_AWAY', 'TEAM_FEATURE_PTS_TOTAL_per_min_ewma_COURT_AWAY', 'TEAM_FEATURE_PTS_TOTAL_cumulative_sum_COURT_HOME', 'TEAM_FEATURE_PTS_TOTAL_per_min_expanding_mean_COURT_HOME', 'TEAM_FEATURE_PTS_TOTAL_per_min_ewma_HOME', 'TEAM_FEATURE_PTS_TOTAL_per_min_expanding_mean_HOME', 'TEAM_FEATURE_cumulative_pt_pct_COURT_AWAY', 'TEAM_FEATURE_PTS_TOTAL_cumulative_sum_COURT_AWAY',
'implied_prob_home',
'implied_prob_away',
'adj_implied_prob_home',
'GAME_DATE_year',
'GAME_DATE_week_number'
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
    'seed':[29],
    'train_frac':[0.8],
    'bet_dist_type':['opt'],
    'bet_budget':[100]
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

train_data_init = pd.read_csv(processed_dir / 'training_data.csv', in_delim)


##################
# test model training
##################

# prep
modeling_round_var = ['SEASON_NUMBER']
bet_round_var = ['GAME_DATE_year','GAME_DATE_week_number']

rounds = train_data_init.groupby(modeling_round_var + bet_round_var).size().reset_index().drop(0, axis=1)

start_ind = 204
current_modeling_round = rounds.loc[start_ind,modeling_round_var][0]
recommender = None
cumulative_profit = 0
cumulative_correct_bets = 0
cumulative_bets = 0
stop_ind = 227

for j in range(start_ind, stop_ind):
    test_round = rounds.iloc[j:j+1,:].reset_index(drop=True)

    if recommender is None or test_round.loc[0,modeling_round_var][0] != current_modeling_round:
        train_rounds = rounds.iloc[0:j,:].reset_index(drop=True)
        train_dat = train_data_init.merge(train_rounds, how='inner', on=modeling_round_var+bet_round_var).copy()

        recommender = Recommender(parameters=model_params, training_data=train_dat, training_function=train_xgb_classifier,  prediction_function=predict_with_xgb_classifier, model_performance_function=calc_model_performance, predictors=predictors, target=target, bet_recommendation_function=calc_bet_distribution, bet_performance_function=calc_total_profit)
        recommender.train_model()
        current_modeling_round = test_round.loc[0,modeling_round_var][0]

    test_dat = train_data_init.merge(test_round, how='inner', on=modeling_round_var+bet_round_var).copy()
    recommender.predict_and_assign_home_away_probabilities(test_dat)
    recommender.recommend_bets(test_dat)
    calc_expected_profit(test_dat)
    results = recommender.calc_performance(test_dat, type='both')
    cumulative_profit = cumulative_profit + results['bet']['total_profit'][0]
    cumulative_correct_bets = cumulative_correct_bets + results['bet']['total_correct_bets'][0]
    cumulative_bets = cumulative_bets + results['bet']['total_bets_placed'][0]
    print(cumulative_profit)
    print(cumulative_correct_bets)


mod = recommender.modeler.model_object
plot_xgb_feat_importance(mod.base_estimator, predictors, 'gain', 'blue', 30)


#to do - build loop for train/test
# to do - add cumulative profit and results tracker

