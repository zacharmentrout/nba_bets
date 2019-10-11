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
import featexp
from joblib import delayed
from joblib import Parallel

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


# constants
ODDS_COL_HOME = 'amax_odds_dec_home'
ODDS_COL_AWAY = 'amax_odds_dec_away'


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
    #fig = plt.figure(dpi=180)
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
    test_data['profit_home'] = test_data['bet_home'] * (test_data[ODDS_COL_HOME] * test_data['place_bet_home'] * test_data['WIN_HOME'] - 1)
    test_data['profit_away'] = test_data['bet_away'] * (test_data[ODDS_COL_AWAY] * test_data['place_bet_away'] * (1-test_data['WIN_HOME']) - 1)
    test_data['profit'] = test_data['profit_home'] + test_data['profit_away']

    test_data['correct_bet_home'] = test_data['place_bet_home'] * test_data['WIN_HOME']
    test_data['correct_bet_away'] = test_data['place_bet_away'] * (1 - test_data['WIN_HOME'])
    test_data['correct_bet'] = test_data['correct_bet_home'] + test_data['correct_bet_away']

    out['total_profit'] = [test_data['profit'].sum()]
    out['total_exp_profit'] = [test_data['exp_profit'].sum()]
    out['sd_profit'] = sqrt(variance_cumulative_profit(test_data))
    out['total_bets_placed'] = [test_data['place_bet'].sum()]
    out['total_correct_bets'] = [test_data['correct_bet'].sum()]
    out['correct_bet_pct'] = out['total_correct_bets'] / out['total_bets_placed']
    out['total_games'] = [test_data.shape[0]]
    out['total_available_bets'] = [test_data.shape[0] - test_data[ODDS_COL_HOME].isna().sum()]
    out['total_bet_amount_home'] = test_data['bet_home'].sum()
    out['total_bet_amount_away'] = test_data['bet_away'].sum()
    out['total_bet_amount'] = [out['total_bet_amount_home'] + out['total_bet_amount_away']]
    out['max_profit'] = test_data['profit'].max()
    out['min_profit'] = test_data['profit'].min()
    out['sharpe_ratio_profit'] = out['total_exp_profit'] / out['sd_profit']
    out['total_exp_roi'] = [out['total_exp_profit'] / out['total_bet_amount']]
    out['roi'] = out['total_profit'] / out['total_bet_amount']
    return(out)


def indicate(x):
    res = [1 if z > 0 else 0 for z in x]
    return(res)


def calc_bet_distribution(test_data, **kwargs):

    if 'conf_threshold' in kwargs.keys():
        conf_threshold = kwargs['conf_threshold']
    else:
        conf_threshold = 0

    if 'min_exp_roi' in kwargs.keys():
        min_exp_roi = kwargs['min_exp_roi']
    else:
        min_exp_roi = 0

    test_data['positive_expectation_HOME'] = indicate(test_data['prob_WIN_HOME'] - 1/test_data[ODDS_COL_HOME])
    test_data['positive_expectation_AWAY'] = indicate((1 - test_data['prob_WIN_HOME']) - 1/test_data[ODDS_COL_AWAY])

    if 'bet_dist_type' in kwargs.keys():
        bet_dist_type = kwargs['bet_dist_type']
    else:
        bet_dist_type = 'unif'

    if 'bet_budget' in kwargs.keys():
        bet_budget = kwargs['bet_budget']
    else:
        bet_budget = 1

    if bet_dist_type == 'opt':
        max_sharpe_ratio(test_data, conf_threshold, min_exp_roi)
        test_data['bet_home'] = test_data['bet_budget_pct_home'] * bet_budget
        test_data['bet_away'] = test_data['bet_budget_pct_away'] * bet_budget
        test_data['place_bet_home'] = (test_data['bet_home'] > 0.001)*1
        test_data['place_bet_away'] = (test_data['bet_away'] > 0.001)*1
        test_data['place_bet'] = test_data['place_bet_home'] + test_data['place_bet_away']
        return(test_data)


    if bet_dist_type == 'max-ep':
        test_data['calc_exp_profit_home'] = test_data['prob_WIN_HOME'] * test_data[ODDS_COL_HOME] - 1
        test_data['calc_exp_profit_away'] = (1 - test_data['prob_WIN_HOME']) * test_data[ODDS_COL_AWAY] - 1
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
    def __init__(self, parameters, bet_recommendation_function, performance_function, odds_home_col=ODDS_COL_HOME, odds_away_col=ODDS_COL_AWAY, bet_home_col='place_bet_HOME', bet_away_col='place_bet_AWAY'):
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
    def __init__(self, parameters, training_data, training_function,  prediction_function, model_performance_function, predictors, target, bet_recommendation_function, bet_performance_function, prediction_col='pred', prob_home_col='prob_WIN_HOME', prob_away_col='prob_WIN_AWAY', odds_home_col=ODDS_COL_HOME, odds_away_col=ODDS_COL_AWAY, bet_home_col='bet_home', bet_away_col='bet_away'):

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

class Simulation:
    def __init__(self, parameters, training_data, predictors, target):

        self.parameters = parameters
        self.modeling_round_var = parameters['modeling_round_var']
        self.betting_round_var = parameters['betting_round_params'][0]
        self.start_betting_round = parameters['betting_round_params'][1]
        self.end_betting_round = parameters['betting_round_params'][2]
        self.training_data = training_data
        self.training_function = parameters['training_prediction_functions'][0]
        self.prediction_function = parameters['training_prediction_functions'][1]
        self.model_performance_function = parameters['model_performance_function']
        self.bet_recommendation_function = parameters['bet_recommendation_function']
        self.bet_performance_function = parameters['bet_performance_function']

        self.sim_results = []

        self.predictors = predictors
        self.target = target
        self.cumulative_profit = [0]
        self.recommender = None


        self.rounds = self.training_data.groupby(self.modeling_round_var + self.betting_round_var).size().reset_index().drop(0, axis=1)

        self.start_ind = np.where(np.all(self.rounds[self.betting_round_var].values == self.start_betting_round,axis=1))[0][0]
        self.stop_ind = np.where(np.all(self.rounds[self.betting_round_var].values == self.end_betting_round,axis=1))[0][0]+1

        self.current_modeling_round = self.rounds.loc[self.start_ind,self.modeling_round_var][0]

    def sim(self):

        for j in range(self.start_ind, self.stop_ind):
            test_round = self.rounds.iloc[j:j+1,:].reset_index(drop=True)

            if self.recommender is None or test_round.loc[0,self.modeling_round_var][0] != self.current_modeling_round:
                train_rounds = self.rounds.iloc[0:j,:].reset_index(drop=True)
                train_dat = self.training_data.merge(train_rounds, how='inner', on=self.modeling_round_var+self.betting_round_var).copy()

                self.recommender = Recommender(parameters=self.parameters, training_data=train_dat, training_function=self.training_function,  prediction_function=self.prediction_function, model_performance_function=self.model_performance_function, predictors=self.predictors, target=self.target, bet_recommendation_function=self.bet_recommendation_function, bet_performance_function=self.bet_performance_function)
                self.recommender.train_model()
                self.current_modeling_round = test_round.loc[0,self.modeling_round_var][0]

            test_dat = self.training_data.merge(test_round, how='inner', on=self.modeling_round_var+self.betting_round_var).copy()
            self.recommender.predict_and_assign_home_away_probabilities(test_dat)
            self.recommender.recommend_bets(test_dat)
            calc_expected_profit(test_dat)
            results = self.recommender.calc_performance(test_dat, type='both')

            bet_results_dict = dict(zip(results['bet'].columns, results['bet'].iloc[0].values))
            model_results_dict = dict(zip(results['model'].columns, results['model'].iloc[0].values))
            self.sim_results.append({**bet_results_dict, **model_results_dict})

            self.cumulative_profit.append(self.cumulative_profit[-1] + results['bet']['total_profit'][0])

        self.sim_results = pd.concat([self.rounds.iloc[self.start_ind:self.stop_ind].reset_index(drop=True),pd.DataFrame(self.sim_results).reset_index(drop=True)], axis=1)

        param_df = pd.DataFrame([format_dict_for_dataframe(self.parameters)])
        self.sim_results = pd.concat([pd.concat([param_df]*(self.sim_results.shape[0])).reset_index(drop=True), self.sim_results.reset_index(drop=True)], axis=1)

def convert_dict_value_for_dataframe(v):
    if callable(v):
        return v.__name__
    if isinstance(v, list) or isinstance(v, tuple):
        return str(v)
    return v

def format_dict_for_dataframe(d):
    return dict(zip(d.keys(), [convert_dict_value_for_dataframe(v) for v in d.values()]))

def calc_expected_profit(test_data):
    test_data['exp_profit_home'] = (test_data['prob_WIN_HOME'] * test_data[ODDS_COL_HOME] - 1) * test_data['bet_home']
    test_data['exp_profit_away'] = (test_data['prob_WIN_AWAY'] * test_data[ODDS_COL_AWAY] - 1) * test_data['bet_away']
    test_data['exp_profit'] = test_data['exp_profit_home'] + test_data['exp_profit_away']
    return(test_data)

def expected_cumulative_profit2(b, p, o):
    result = sum((p * o - 1) * b)
    return(result)

def expected_cumulative_profit(test_data):
    test_data = calc_expected_profit(test_data)
    return(test_data['exp_profit'].sum())

def variance_cumulative_profit(test_data):
    sum1 = ((1 - test_data['prob_WIN_HOME']) * test_data['prob_WIN_HOME'] * pow(test_data['bet_home'], 2) * pow(test_data[ODDS_COL_HOME], 2)).sum()
    sum2 = ((1 - test_data['prob_WIN_AWAY']) * test_data['prob_WIN_AWAY'] * pow(test_data['bet_away'], 2) * pow(test_data[ODDS_COL_AWAY], 2)).sum()
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

def max_sharpe_ratio(test_data, conf_threshold=0, min_exp_roi=0):
    dat_copy = test_data.copy()
    valid_rows_conf = dat_copy.index[[False if (x) or (abs(y-0.5) <= conf_threshold ) else True for x,y in zip(dat_copy[ODDS_COL_HOME].isna(), dat_copy['prob_WIN_HOME']) ]].tolist()

    valid_rows_exp_roi = dat_copy.index[[False if (x) or ((y1*z1 - 1 <= min_exp_roi) and (y2*z2 - 1 <= min_exp_roi)) else True for x,y1,z1,y2,z2 in zip(dat_copy[ODDS_COL_HOME].isna(), dat_copy['prob_WIN_HOME'], dat_copy[ODDS_COL_HOME], dat_copy['prob_WIN_AWAY'], dat_copy[ODDS_COL_AWAY]) ]].tolist()

    valid_rows = list(set(valid_rows_conf).intersection(set(valid_rows_exp_roi)))

    if len(valid_rows) == 0:
        test_data['bet_budget_pct_home'] = 0
        test_data['bet_budget_pct_away'] = 0
        return(test_data)

    dat_copy = dat_copy.loc[valid_rows]

    p = dat_copy['prob_WIN_HOME'].append(dat_copy['prob_WIN_AWAY'])
    o = dat_copy[ODDS_COL_HOME].append(dat_copy[ODDS_COL_AWAY])
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
# 'TEAM_FEATURE_PF_per_min_expanding_mean_AWAY',
# 'TEAM_FEATURE_PF_per_min_expanding_mean_HOME',
# 'TEAM_FEATURE_FG3A_per_min_expanding_mean_AWAY',
# 'TEAM_FEATURE_FG3A_per_min_expanding_mean_HOME',
# 'TEAM_FEATURE_FGA_per_min_expanding_mean_AWAY',
# 'TEAM_FEATURE_FGA_per_min_expanding_mean_HOME',
# 'TEAM_FEATURE_DREB_per_min_expanding_mean_AWAY',
# 'TEAM_FEATURE_DREB_per_min_expanding_mean_HOME',
# 'TEAM_FEATURE_REB_per_min_expanding_mean_AWAY',
# 'TEAM_FEATURE_REB_per_min_expanding_mean_HOME',
# 'TEAM_FEATURE_BLK_per_min_expanding_mean_AWAY',
# 'TEAM_FEATURE_BLK_per_min_expanding_mean_HOME',
# 'TEAM_FEATURE_AST_per_min_expanding_mean_AWAY',
# 'TEAM_FEATURE_AST_per_min_expanding_mean_HOME',
# 'TEAM_FEATURE_OREB_per_min_expanding_mean_AWAY',
# 'TEAM_FEATURE_OREB_per_min_expanding_mean_HOME',
# 'TEAM_FEATURE_FG3M_per_min_expanding_mean_AWAY',
# 'TEAM_FEATURE_FG3M_per_min_expanding_mean_HOME',
# 'TEAM_FEATURE_STL_per_min_expanding_mean_AWAY',
# 'TEAM_FEATURE_STL_per_min_expanding_mean_HOME',
# 'TEAM_FEATURE_FGM_per_min_expanding_mean_AWAY',
# 'TEAM_FEATURE_FGM_per_min_expanding_mean_HOME',
# 'TEAM_FEATURE_PTS_per_min_expanding_mean_AWAY',
# 'TEAM_FEATURE_PTS_per_min_expanding_mean_HOME',
# 'TEAM_FEATURE_PTS_TOTAL_per_min_expanding_mean_AWAY',
# 'TEAM_FEATURE_PTS_TOTAL_per_min_expanding_mean_HOME',
# 'TEAM_FEATURE_FTM_per_min_expanding_mean_AWAY',
# 'TEAM_FEATURE_FTM_per_min_expanding_mean_HOME',
# 'TEAM_FEATURE_FG_PCT_expanding_mean_AWAY',
# 'TEAM_FEATURE_FG_PCT_expanding_mean_HOME',
# 'TEAM_FEATURE_FG3_PCT_expanding_mean_AWAY',
# 'TEAM_FEATURE_FG3_PCT_expanding_mean_HOME',
# 'TEAM_FEATURE_FT_PCT_expanding_mean_AWAY',
# 'TEAM_FEATURE_FT_PCT_expanding_mean_HOME',
# 'TEAM_FEATURE_OFF_RATING_expanding_mean_AWAY',
# 'TEAM_FEATURE_OFF_RATING_expanding_mean_HOME',
# 'TEAM_FEATURE_DEF_RATING_expanding_mean_AWAY',
# 'TEAM_FEATURE_DEF_RATING_expanding_mean_HOME',
# 'TEAM_FEATURE_EFG_PCT_expanding_mean_AWAY',
# 'TEAM_FEATURE_EFG_PCT_expanding_mean_HOME',
# 'TEAM_FEATURE_TM_TOV_PCT_expanding_mean_AWAY',
# 'TEAM_FEATURE_TM_TOV_PCT_expanding_mean_HOME',
# 'TEAM_FEATURE_PF_per_min_ewma_AWAY',
# 'TEAM_FEATURE_PF_per_min_ewma_HOME',
# 'TEAM_FEATURE_FG3A_per_min_ewma_AWAY',
# 'TEAM_FEATURE_FG3A_per_min_ewma_HOME',
# 'TEAM_FEATURE_FGA_per_min_ewma_AWAY',
# 'TEAM_FEATURE_FGA_per_min_ewma_HOME',
# 'TEAM_FEATURE_DREB_per_min_ewma_AWAY',
# 'TEAM_FEATURE_DREB_per_min_ewma_HOME',
# 'TEAM_FEATURE_REB_per_min_ewma_AWAY',
# 'TEAM_FEATURE_REB_per_min_ewma_HOME',
# 'TEAM_FEATURE_BLK_per_min_ewma_AWAY',
# 'TEAM_FEATURE_BLK_per_min_ewma_HOME',
# 'TEAM_FEATURE_AST_per_min_ewma_AWAY',
# 'TEAM_FEATURE_AST_per_min_ewma_HOME',
# 'TEAM_FEATURE_OREB_per_min_ewma_AWAY',
# 'TEAM_FEATURE_OREB_per_min_ewma_HOME',
# 'TEAM_FEATURE_FG3M_per_min_ewma_AWAY',
# 'TEAM_FEATURE_FG3M_per_min_ewma_HOME',
# 'TEAM_FEATURE_STL_per_min_ewma_AWAY',
# 'TEAM_FEATURE_STL_per_min_ewma_HOME',
# 'TEAM_FEATURE_FGM_per_min_ewma_AWAY',
# 'TEAM_FEATURE_FGM_per_min_ewma_HOME',
# 'TEAM_FEATURE_PTS_per_min_ewma_AWAY',
# 'TEAM_FEATURE_PTS_per_min_ewma_HOME',
# 'TEAM_FEATURE_PTS_TOTAL_per_min_ewma_AWAY',
# 'TEAM_FEATURE_PTS_TOTAL_per_min_ewma_HOME',
# 'TEAM_FEATURE_FTM_per_min_ewma_AWAY',
# 'TEAM_FEATURE_FTM_per_min_ewma_HOME',
# 'TEAM_FEATURE_FG_PCT_ewma_AWAY',
# 'TEAM_FEATURE_FG_PCT_ewma_HOME',
# 'TEAM_FEATURE_FG3_PCT_ewma_AWAY',
# 'TEAM_FEATURE_FG3_PCT_ewma_HOME',
# 'TEAM_FEATURE_FT_PCT_ewma_AWAY',
# 'TEAM_FEATURE_FT_PCT_ewma_HOME',
'TEAM_FEATURE_OFF_RATING_ewma_AWAY',
'TEAM_FEATURE_OFF_RATING_ewma_HOME',
'TEAM_FEATURE_DEF_RATING_ewma_AWAY',
'TEAM_FEATURE_DEF_RATING_ewma_HOME',
# 'TEAM_FEATURE_EFG_PCT_ewma_AWAY',
# 'TEAM_FEATURE_EFG_PCT_ewma_HOME',
# 'TEAM_FEATURE_TM_TOV_PCT_ewma_AWAY',
# 'TEAM_FEATURE_TM_TOV_PCT_ewma_HOME',
# 'TEAM_FEATURE_cumulative_win_pct_AWAY',
# 'TEAM_FEATURE_cumulative_win_pct_HOME',
# 'TEAM_FEATURE_EFG_PCT_cumulative_AWAY',
# 'TEAM_FEATURE_EFG_PCT_cumulative_HOME',
# 'TEAM_FEATURE_TM_TOV_PCT_cumulative_AWAY',
# 'TEAM_FEATURE_TM_TOV_PCT_cumulative_HOME',
'TEAM_FEATURE_OFF_RATING_ewma_pythag_AWAY',
'TEAM_FEATURE_OFF_RATING_ewma_pythag_HOME',
# 'TEAM_FEATURE_PF_per_min_expanding_mean_COURT_AWAY',
# 'TEAM_FEATURE_PF_per_min_expanding_mean_COURT_HOME',
# 'TEAM_FEATURE_FG3A_per_min_expanding_mean_COURT_AWAY',
# 'TEAM_FEATURE_FG3A_per_min_expanding_mean_COURT_HOME',
# 'TEAM_FEATURE_FGA_per_min_expanding_mean_COURT_AWAY',
# 'TEAM_FEATURE_FGA_per_min_expanding_mean_COURT_HOME',
# 'TEAM_FEATURE_DREB_per_min_expanding_mean_COURT_AWAY',
# 'TEAM_FEATURE_DREB_per_min_expanding_mean_COURT_HOME',
# 'TEAM_FEATURE_REB_per_min_expanding_mean_COURT_AWAY',
# 'TEAM_FEATURE_REB_per_min_expanding_mean_COURT_HOME',
# 'TEAM_FEATURE_BLK_per_min_expanding_mean_COURT_AWAY',
# 'TEAM_FEATURE_BLK_per_min_expanding_mean_COURT_HOME',
# 'TEAM_FEATURE_AST_per_min_expanding_mean_COURT_AWAY',
# 'TEAM_FEATURE_AST_per_min_expanding_mean_COURT_HOME',
# 'TEAM_FEATURE_OREB_per_min_expanding_mean_COURT_AWAY',
# 'TEAM_FEATURE_OREB_per_min_expanding_mean_COURT_HOME',
# 'TEAM_FEATURE_FG3M_per_min_expanding_mean_COURT_AWAY',
# 'TEAM_FEATURE_FG3M_per_min_expanding_mean_COURT_HOME',
# 'TEAM_FEATURE_STL_per_min_expanding_mean_COURT_AWAY',
# 'TEAM_FEATURE_STL_per_min_expanding_mean_COURT_HOME',
# 'TEAM_FEATURE_FGM_per_min_expanding_mean_COURT_AWAY',
# 'TEAM_FEATURE_FGM_per_min_expanding_mean_COURT_HOME',
# 'TEAM_FEATURE_PTS_per_min_expanding_mean_COURT_AWAY',
# 'TEAM_FEATURE_PTS_per_min_expanding_mean_COURT_HOME',
# 'TEAM_FEATURE_PTS_TOTAL_per_min_expanding_mean_COURT_AWAY',
# 'TEAM_FEATURE_PTS_TOTAL_per_min_expanding_mean_COURT_HOME',
# 'TEAM_FEATURE_FTM_per_min_expanding_mean_COURT_AWAY',
# 'TEAM_FEATURE_FTM_per_min_expanding_mean_COURT_HOME',
# 'TEAM_FEATURE_FG_PCT_expanding_mean_COURT_AWAY',
# 'TEAM_FEATURE_FG_PCT_expanding_mean_COURT_HOME',
# 'TEAM_FEATURE_FG3_PCT_expanding_mean_COURT_AWAY',
# 'TEAM_FEATURE_FG3_PCT_expanding_mean_COURT_HOME',
# 'TEAM_FEATURE_FT_PCT_expanding_mean_COURT_AWAY',
# 'TEAM_FEATURE_FT_PCT_expanding_mean_COURT_HOME',
# 'TEAM_FEATURE_OFF_RATING_expanding_mean_COURT_AWAY',
# 'TEAM_FEATURE_OFF_RATING_expanding_mean_COURT_HOME',
# 'TEAM_FEATURE_DEF_RATING_expanding_mean_COURT_AWAY',
# 'TEAM_FEATURE_DEF_RATING_expanding_mean_COURT_HOME',
# 'TEAM_FEATURE_EFG_PCT_expanding_mean_COURT_AWAY',
# 'TEAM_FEATURE_EFG_PCT_expanding_mean_COURT_HOME',
# 'TEAM_FEATURE_TM_TOV_PCT_expanding_mean_COURT_AWAY',
# 'TEAM_FEATURE_TM_TOV_PCT_expanding_mean_COURT_HOME',
# 'TEAM_FEATURE_PF_per_min_ewma_COURT_AWAY',
# 'TEAM_FEATURE_PF_per_min_ewma_COURT_HOME',
# 'TEAM_FEATURE_FG3A_per_min_ewma_COURT_AWAY',
# 'TEAM_FEATURE_FG3A_per_min_ewma_COURT_HOME',
# 'TEAM_FEATURE_FGA_per_min_ewma_COURT_AWAY',
# 'TEAM_FEATURE_FGA_per_min_ewma_COURT_HOME',
# 'TEAM_FEATURE_DREB_per_min_ewma_COURT_AWAY',
# 'TEAM_FEATURE_DREB_per_min_ewma_COURT_HOME',
# 'TEAM_FEATURE_REB_per_min_ewma_COURT_AWAY',
# 'TEAM_FEATURE_REB_per_min_ewma_COURT_HOME',
# 'TEAM_FEATURE_BLK_per_min_ewma_COURT_AWAY',
# 'TEAM_FEATURE_BLK_per_min_ewma_COURT_HOME',
# 'TEAM_FEATURE_AST_per_min_ewma_COURT_AWAY',
# 'TEAM_FEATURE_AST_per_min_ewma_COURT_HOME',
# 'TEAM_FEATURE_OREB_per_min_ewma_COURT_AWAY',
# 'TEAM_FEATURE_OREB_per_min_ewma_COURT_HOME',
# 'TEAM_FEATURE_FG3M_per_min_ewma_COURT_AWAY',
# 'TEAM_FEATURE_FG3M_per_min_ewma_COURT_HOME',
# 'TEAM_FEATURE_STL_per_min_ewma_COURT_AWAY',
# 'TEAM_FEATURE_STL_per_min_ewma_COURT_HOME',
# 'TEAM_FEATURE_FGM_per_min_ewma_COURT_AWAY',
# 'TEAM_FEATURE_FGM_per_min_ewma_COURT_HOME',
# 'TEAM_FEATURE_PTS_per_min_ewma_COURT_AWAY',
# 'TEAM_FEATURE_PTS_per_min_ewma_COURT_HOME',
# 'TEAM_FEATURE_PTS_TOTAL_per_min_ewma_COURT_AWAY',
# 'TEAM_FEATURE_PTS_TOTAL_per_min_ewma_COURT_HOME',
# 'TEAM_FEATURE_FTM_per_min_ewma_COURT_AWAY',
# 'TEAM_FEATURE_FTM_per_min_ewma_COURT_HOME',
# 'TEAM_FEATURE_FG_PCT_ewma_COURT_AWAY',
# 'TEAM_FEATURE_FG_PCT_ewma_COURT_HOME',
# 'TEAM_FEATURE_FG3_PCT_ewma_COURT_AWAY',
# 'TEAM_FEATURE_FG3_PCT_ewma_COURT_HOME',
# 'TEAM_FEATURE_FT_PCT_ewma_COURT_AWAY',
# 'TEAM_FEATURE_FT_PCT_ewma_COURT_HOME',
'TEAM_FEATURE_OFF_RATING_ewma_COURT_AWAY',
'TEAM_FEATURE_OFF_RATING_ewma_COURT_HOME',
'TEAM_FEATURE_DEF_RATING_ewma_COURT_AWAY',
'TEAM_FEATURE_DEF_RATING_ewma_COURT_HOME',
'TEAM_FEATURE_EFG_PCT_ewma_COURT_AWAY',
'TEAM_FEATURE_EFG_PCT_ewma_COURT_HOME',
'TEAM_FEATURE_TM_TOV_PCT_ewma_COURT_AWAY',
'TEAM_FEATURE_TM_TOV_PCT_ewma_COURT_HOME',
'TEAM_FEATURE_cumulative_win_pct_COURT_AWAY',
'TEAM_FEATURE_cumulative_win_pct_COURT_HOME',
'TEAM_FEATURE_cumulative_pt_pct_COURT_AWAY',
'TEAM_FEATURE_cumulative_pt_pct_COURT_HOME',
'TEAM_FEATURE_TM_TOV_PCT_cumulative_COURT_AWAY',
'TEAM_FEATURE_TM_TOV_PCT_cumulative_COURT_HOME',
'TEAM_FEATURE_EFG_PCT_cumulative_COURT_AWAY',
'TEAM_FEATURE_EFG_PCT_cumulative_COURT_HOME',
'TEAM_FEATURE_OFF_RATING_ewma_pythag_COURT_AWAY',
'TEAM_FEATURE_OFF_RATING_ewma_pythag_COURT_HOME',

'TEAM_FEATURE_OFF_RATING_ewma_pythag_HOME_adj',
'TEAM_FEATURE_OFF_RATING_ewma_pythag_AWAY_adj',

'TEAM_FEATURE_OFF_RATING_ewma_HOME_adj',
'TEAM_FEATURE_OFF_RATING_ewma_AWAY_adj',

# 'implied_prob_home',
# 'implied_prob_away',
# 'adj_implied_prob_home',
'GAME_DATE_year',
'GAME_DATE_week_number'
]

predictors = [
'TEAM_FEATURE_cumulative_win_pct_COURT_AWAY',
'TEAM_FEATURE_cumulative_win_pct_COURT_HOME',
'TEAM_FEATURE_cumulative_pt_pct_COURT_AWAY',
'TEAM_FEATURE_cumulative_pt_pct_COURT_HOME',
'TEAM_FEATURE_OFF_RATING_ewma_pythag_HOME_adj',
'TEAM_FEATURE_OFF_RATING_ewma_pythag_AWAY_adj',
'TEAM_FEATURE_cumulative_pt_pct_COURT_HOME_AWAY_diff',
'TEAM_FEATURE_cumulative_pt_pct_COURT_HOME_AWAY_frac',
'TEAM_FEATURE_cumulative_win_pct_COURT_HOME_AWAY_diff',
'TEAM_FEATURE_cumulative_win_pct_COURT_HOME_AWAY_frac',
#'TEAM_FEATURE_cumulative_pt_pct_COURT_pythag_HOME',
#'TEAM_FEATURE_cumulative_win_pct_COURT_pythag_HOME'
#'implied_prob_home',
#'implied_prob_away',
]

model_param_dict = {
    'learning_rate':[0.01],
    'n_estimators':[10],
    'max_depth':[3],
    'min_child_weight':[1],
    'gamma':[0.5],
    'subsample':[1],
    'colsample_bytree':[1.0],
    'objective':['binary:logistic'],
    'nthread':[1],
    'scale_pos_weight':[1],
    'seed':[4445, 9992, 1, 23, 55],
    'train_frac':[0.5, 0.75],
    'bet_dist_type':['max-ep'],
    'bet_budget':[100],
    'min_exp_roi':[0.3],
    #'conf_threshold':[0.0]

    # functions
    'training_prediction_functions':[(train_xgb_classifier, predict_with_xgb_classifier)],
    'model_performance_function':[calc_model_performance],
    'bet_recommendation_function':[calc_bet_distribution],
    'bet_performance_function':[calc_total_profit],
    'modeling_round_var':[['SEASON_NUMBER']],
    'betting_round_params':[(['GAME_DATE_year','GAME_DATE_week_number'], [2016, 46], [2017, 15])]
}


#model_param_row_number = 0

##################
# get model parameters
##################
model_param_names = sorted(model_param_dict)
model_param_combinations = it.product(*(model_param_dict[Name] for Name in model_param_names))
model_param_list = list(model_param_combinations)
model_param_df = pd.DataFrame(model_param_list, columns=model_param_names)
model_params = [{key:val for key, val in zip(model_param_names, model_param_list[i])} for i in range(model_param_df.shape[0])]

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


class Simulator:
    def __init__(self, param_dict_list, training_data, predictors, target):
        self.param_dict_list = param_dict_list
        self.training_data = training_data
        self.predictors = predictors
        self.target = target

        self.simulations = None

    def simulate(self, max_parallel_cpus=-1):
        self.simulations = Parallel(n_jobs=max_parallel_cpus)(delayed(self.run_simulation)(param_dict) for param_dict in self.param_dict_list)

    def run_simulation(self, parameters):
        simulation = Simulation(parameters, training_data=self.training_data, predictors=self.predictors, target=self.target)
        simulation.sim()
        return(simulation)

    def simulation_output(self):
        if (self.simulations is None):
            return []
        return pd.concat([s.sim_results for s in self.simulations]).reset_index(drop=True)


agg_param_names = list(set(model_param_dict.keys()) - set(['seed']))

simulator = Simulator(param_dict_list=model_params, training_data=train_data_init, predictors=predictors, target=target)

# todo use function names in model param dict and grab functions later in constructor with locals()['function_name']
simulator.simulate()
sim_out = simulator.simulation_output()

# figure out window function for sim_out
sim_out['param_num'] = sim_out[agg_param_names].apply(lambda x: reduce(lambda a, b: str(a) + str(b), x), axis=1).rank(method='dense', ascending=True)

mod = simulation.recommender.modeler.model_object
plot_xgb_feat_importance(mod.base_estimator, predictors, 'gain', 'blue', 30)



train1 = train_data_init[train_data_init['SEASON_NUMBER']<=15]
test1 = train_data_init[train_data_init['SEASON_NUMBER']>15]


featexp.get_univariate_plots(data=train_dat, target_col=target,
                     features_list=['TEAM_FEATURE_cumulative_pt_pct_COURT_AWAY'], bins=10)

featexp.get_univariate_plots(data=train1, target_col=target, data_test=test1, features_list=['TEAM_FEATURE_cumulative_pt_pct_pythag_COURT_HOME'])

featexp.get_univariate_plots(data=train1, target_col=target, data_test=test1, features_list=['TEAM_FEATURE_OFF_RATING_ewma_pythag_AWAY_adj'])

featexp.get_univariate_plots(data=train1, target_col=target, data_test=test1, features_list=predictors)

stats = featexp.get_trend_stats(data=train1[predictors+[target]], target_col=target, data_test=test1[predictors+[target]])


stats.to_csv(processed_dir / 'featexp_stats.csv', index=False, sep=',')

stats_keep = stats[(stats['Trend_changes'] == 0) & (stats['Trend_changes_test'] == 0) & (stats['Trend_correlation'] >= 0.97)]
# to do - build loop for train/test
# to do - add cumulative profit and results tracker

