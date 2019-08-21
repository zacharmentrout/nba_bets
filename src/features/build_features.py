#!/usr/bin/env python3
# -*- coding:  utf-8 -*-
"""
Created on Wed Aug 14 2019

@author:  zach
"""
import pandas as pd
import os
import sklearn as sk
import math
import re
from pathlib import Path
import numpy as np
from datetime import datetime
from dateutil.parser import parse
import featuretools as ft
import copy


def odds2prob_ml(odds):
    if math.isnan(odds):
        return(float('nan'))
    if odds < 0:
        return(-odds / (-odds + 100))
    return(100 / (odds + 100))


def quartile1(x):
    return (np.nanpercentile(x, 25))


def quartile3(x):
    return (np.nanpercentile(x, 75))


def multiple_replace(dict,  text):
    # Create a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape,  dict.keys())))
    # For each match,  look-up corresponding value in dictionary
    return regex.sub(lambda mo:  dict[mo.string[mo.start(): mo.end()]],  text)

def calc_stat_per_minute(dat, col_name_stat, col_name_minutes='MIN'):
    new_col = dat[col_name_stat] / dat[col_name_minutes]
    return(new_col)

def calc_groupby_feature(df, apply_fn, apply_cols, groupby_col, suffix='', prefix=''):
    results_list = [apply_fn(df, c, groupby_col) for c in apply_cols]
    results_df = pd.DataFrame(list(zip(*results_list)), columns=[prefix + s + suffix for s in apply_cols])
    df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
    return(df)

def calc_stat_cumsum(df, apply_col, groupby_col):
    #return(df.groupby(groupby_col)[apply_col].cumsum())
    return(df.groupby(groupby_col)[apply_col].apply(lambda x:  x.shift(1).cumsum()))

def calc_stat_expanding_mean(df, apply_col, groupby_col):
    #return(df.groupby(groupby_col)[apply_col].expanding().mean())
    return(df.groupby(groupby_col)[apply_col].apply(lambda x: x.shift(1).expanding().mean()))

def calc_stat_ewma(df, apply_col, groupby_col):
    return(df.groupby(groupby_col)[apply_col].apply(lambda x: x.shift(1).ewm(halflife=5).mean()))

def left(s, amount):
    return s[:amount]

def right(s, amount):
    return s[-amount:]

def make_court_col(df, s):
    result = [x if ~np.isnan(x) else y for x,y in zip(df[s+'HOME'], df[s+'AWAY'])]
    result_df = pd.DataFrame(result, columns=[s+'COURT'])
    return(result_df)


# location of data directory
data_location = '/Users/zach/Documents/git/nba_bets/data'

data_dir = Path(data_location)
raw_dir = data_dir / 'raw'
interim_dir = data_dir / 'interim'
processed_dir = data_dir / 'processed'

# read in data
team_data =  pd.read_csv(raw_dir / 'game_data_by_team.csv', sep = '|')

team_data.sort_values(by=['TEAM_NUMBER',  'GAME_DATE'], inplace=True)

# get game id for next and previous games
team_data['GAME_NUMBER_prev_game'] = team_data.groupby(['TEAM_NUMBER', 'SEASON_NUMBER'])['GAME_NUMBER'].apply(lambda x:  x.shift(1))
team_data['GAME_NUMBER_next_game'] = team_data.groupby(['TEAM_NUMBER', 'SEASON_NUMBER'])['GAME_NUMBER'].apply(lambda x:  x.shift(-1))


# calculate per-minute stats
team_stats_per_min = [
'AST'
,'BLK'
,'DREB'
,'FG PCT'
,'FG3 PCT'
,'FG3A'
,'FG3M'
,'FGA'
,'FGM'
,'FT PCT'
,'FTA '
,'FTM'
,'OREB'
,'PF'
,'PLUS MINUS'
,'PTS'
,'REB'
,'STL'
,'TO'
]

team_stats_per_min =list(set(team_stats_per_min).intersection(set(team_data.columns)))
team_stats_per_min_list = [calc_stat_per_minute(team_data, c) for c in team_stats_per_min]
team_stats_per_min_df = pd.DataFrame(list(zip(*team_stats_per_min_list)),
               columns =[s + '_per_min' for s in team_stats_per_min])

# put it all together
team_data = pd.concat([team_data.reset_index(drop=True), team_stats_per_min_df], axis=1)

# add w/l indicator
team_data['WIN'] = [1 if s == 'W' else 0 for s in team_data['WL']]
team_data['LOSE'] = [1 if s == 'L' else 0 for s in team_data['WL']]


# calculate cumulative features
team_data['TEAM_FEATURE_cumulative_count_GAME_NUMBER'] = team_data.groupby(['TEAM_NUMBER', 'SEASON_NUMBER']).cumcount()

# calculate cumulative sum features
cumsum_features = [
'TEAM_IS_HOME_TEAM',
'TEAM_IS_AWAY_TEAM',
'MIN'
,'WIN'
,'LOSE'
,'PTS'
]

team_data = calc_groupby_feature(team_data, calc_stat_cumsum, cumsum_features, ['TEAM_NUMBER', 'SEASON_NUMBER'], prefix='TEAM_FEATURE_', suffix='_cumulative_sum')


# calculate mean features
mean_features = [s + '_per_min' for s in team_stats_per_min]

team_data = calc_groupby_feature(team_data, calc_stat_expanding_mean, mean_features, ['TEAM_NUMBER', 'SEASON_NUMBER'], prefix='TEAM_FEATURE_', suffix='_expanding_mean')


# calculate exponentially weighted moving avg features
ewma_features = mean_features

team_data=calc_groupby_feature(team_data, calc_stat_ewma, ewma_features, groupby_col= ['TEAM_NUMBER', 'SEASON_NUMBER'], prefix='TEAM_FEATURE_', suffix='_ewma')


# calculate same features for home games vs. away
team_data_home = team_data[team_data['TEAM_IS_HOME_TEAM'] == 1]
team_data_away = team_data[team_data['TEAM_IS_AWAY_TEAM'] == 1]

team_data_home.sort_values(by=['TEAM_NUMBER',  'GAME_DATE'], inplace=True)
team_data_away.sort_values(by=['TEAM_NUMBER',  'GAME_DATE'], inplace=True)


# get game id for next and previous games
team_data_home['GAME_NUMBER_prev_game_HOME'] = team_data_home.groupby(['TEAM_NUMBER', 'SEASON_NUMBER'])['GAME_NUMBER'].apply(lambda x:  x.shift(1))
team_data_home['GAME_NUMBER_next_game_HOME'] = team_data_home.groupby(['TEAM_NUMBER', 'SEASON_NUMBER'])['GAME_NUMBER'].apply(lambda x:  x.shift(-1))

team_data_away['GAME_NUMBER_prev_game_AWAY'] = team_data_away.groupby(['TEAM_NUMBER', 'SEASON_NUMBER'])['GAME_NUMBER'].apply(lambda x:  x.shift(1))
team_data_away['GAME_NUMBER_next_game_AWAY'] = team_data_away.groupby(['TEAM_NUMBER', 'SEASON_NUMBER'])['GAME_NUMBER'].apply(lambda x:  x.shift(-1))

# cumulative sum features
team_data_home = calc_groupby_feature(team_data_home, calc_stat_cumsum, cumsum_features, ['TEAM_NUMBER', 'SEASON_NUMBER'], prefix='TEAM_FEATURE_', suffix='_cumulative_sum_HOME')
team_data_away = calc_groupby_feature(team_data_away, calc_stat_cumsum, cumsum_features, ['TEAM_NUMBER', 'SEASON_NUMBER'], prefix='TEAM_FEATURE_', suffix='_cumulative_sum_AWAY')

# expanding mean features
team_data_home = calc_groupby_feature(team_data_home, calc_stat_expanding_mean, mean_features, ['TEAM_NUMBER', 'SEASON_NUMBER'], prefix='TEAM_FEATURE_', suffix='_expanding_mean_HOME')
team_data_away = calc_groupby_feature(team_data_away, calc_stat_expanding_mean, mean_features, ['TEAM_NUMBER', 'SEASON_NUMBER'], prefix='TEAM_FEATURE_', suffix='_expanding_mean_AWAY')

# ewma features
team_data_home = calc_groupby_feature(team_data_home, calc_stat_ewma, ewma_features, groupby_col= ['TEAM_NUMBER', 'SEASON_NUMBER'], prefix='TEAM_FEATURE_', suffix='_ewma_HOME')
team_data_away= calc_groupby_feature(team_data_away, calc_stat_ewma, ewma_features, groupby_col= ['TEAM_NUMBER', 'SEASON_NUMBER'], prefix='TEAM_FEATURE_', suffix='_ewma_AWAY')


home_cols = [s for s in team_data_home.columns if s.endswith('_HOME') and s.startswith(('TEAM_FEATURE_', 'GAME_NUMBER_'))]
away_cols = [s for s in team_data_away.columns if s.endswith('_AWAY') and s.startswith(('TEAM_FEATURE_', 'GAME_NUMBER_'))]

home_feature_cols = [s for s in home_cols if s.startswith('TEAM_FEATURE')]
away_feature_cols = [s for s in away_cols if s.startswith('TEAM_FEATURE')]

home_away_feature_col_prefixes = [left(s, len(s) - 4) for s in home_feature_cols]

team_data = team_data.merge(team_data_home[['TEAM_NUMBER', 'GAME_NUMBER']+home_cols], how='left', on=['TEAM_NUMBER', 'GAME_NUMBER']).merge(team_data_away[['TEAM_NUMBER', 'GAME_NUMBER']+away_cols], how='left', on=['TEAM_NUMBER', 'GAME_NUMBER'])


court_col_list = [make_court_col(team_data, c) for c in home_away_feature_col_prefixes]

court_col_df = pd.concat(court_col_list, axis=1)

team_data = pd.concat([team_data, court_col_df], axis=1)

team_data.to_csv(raw_dir / 'features_check.csv',  sep=',', index=False)


# pivot on home_vs_away

team_feature_cols = [s for s in team_data.columns if s.startswith('TEAM_FEATURE_') and not s.endswith('_HOME') and not s.endswith('_AWAY')]

team_data_pivot = team_data.pivot(index='GAME_NUMBER', columns='TEAM_HOME_OR_AWAY',  values=team_feature_cols)

team_data_pivot.reset_index(level=0, inplace=True)

old_names = list(team_data_pivot.columns.values)
new_names = [str(o[0])+'_'+str(o[1]) for o in old_names]

team_data_pivot.columns = new_names

team_data_pivot.to_csv(raw_dir / 'features_check2.csv',  sep=',', index=False)


# to do:
# add dim_game data
# add other columns to pivot besides these
