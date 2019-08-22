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

def multiindex_pivot(df, columns=None, values=None):
    #https://github.com/pandas-dev/pandas/issues/23955
    names = list(df.index.names)
    df = df.reset_index()
    list_index = df[names].values
    tuples_index = [tuple(i) for i in list_index] # hashable
    df = df.assign(tuples_index=tuples_index)
    df = df.pivot(index="tuples_index", columns=columns, values=values)
    tuples_index = df.index  # reduced
    index = pd.MultiIndex.from_tuples(tuples_index, names=names)
    df.index = index
    return df


# location of data directory
data_location = '/Users/zach/Documents/git/nba_bets/data'

data_dir = Path(data_location)
raw_dir = data_dir / 'raw'
interim_dir = data_dir / 'interim'
processed_dir = data_dir / 'processed'

# get dtypes
dtypes_get = pd.read_csv(raw_dir / 'dtypes_game_data_by_team.csv',sep='|',dtype='object')
dtypes_dict = dict(zip(dtypes_get['col_name'], dtypes_get['t']))

# read in data
team_data =  pd.read_csv(raw_dir / 'game_data_by_team.csv', sep = '|', dtype=dtypes_dict)


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

# add targets

home_cols = [s for s in team_data_home.columns if s.endswith('_HOME') and s.startswith(('TEAM_FEATURE_', 'GAME_NUMBER_', 'TARGET_'))]
away_cols = [s for s in team_data_away.columns if s.endswith('_AWAY') and s.startswith(('TEAM_FEATURE_', 'GAME_NUMBER_', 'TARGET'))]

home_feature_cols = [s for s in home_cols if s.startswith('TEAM_FEATURE')]
away_feature_cols = [s for s in away_cols if s.startswith('TEAM_FEATURE')]

#home_cols = home_cols + ['PTS']
#away_cols = away_cols + ['PTS']

home_away_feature_col_prefixes = [left(s, len(s) - 4) for s in home_feature_cols]

team_data = team_data.merge(team_data_home[['TEAM_NUMBER', 'GAME_NUMBER']+home_cols], how='left', on=['TEAM_NUMBER', 'GAME_NUMBER']).merge(team_data_away[['TEAM_NUMBER', 'GAME_NUMBER']+away_cols], how='left', on=['TEAM_NUMBER', 'GAME_NUMBER'])

court_col_list = [make_court_col(team_data, c) for c in home_away_feature_col_prefixes]

court_col_df = pd.concat(court_col_list, axis=1)

team_data = pd.concat([team_data, court_col_df], axis=1)

team_data['WIN_HOME'] = team_data['TEAM_IS_HOME_TEAM'] * team_data['WIN'] + team_data['TEAM_IS_AWAY_TEAM'] * team_data['LOSE']
team_data['WIN_AWAY'] = 1 - team_data['WIN_HOME']

team_data.to_csv(raw_dir / 'features_check.csv',  sep=',', index=False)


# pivot on home_vs_away

team_feature_cols = [s for s in team_data.columns if s.startswith('TEAM_FEATURE_') and not s.endswith('_HOME') and not s.endswith('_AWAY')]

other_pivot_cols = ['WIN_HOME', 'WIN_AWAY']

team_data_new = team_data.set_index(['GAME_NUMBER', 'WIN_HOME'], drop=True)

team_data_pivot = multiindex_pivot(team_data_new, columns='TEAM_HOME_OR_AWAY',  values=team_feature_cols)

team_data_pivot.reset_index(inplace=True)

old_names = list(team_data_pivot.columns.values)
new_names = [str(o[0])+'_'+str(o[1]) if o[1] != '' else o[0] for o in old_names]

team_data_pivot.columns = new_names

# add dim_game data
dim_game = pd.read_csv(raw_dir / 'dim_game.csv', sep='|')


# training data full

train_data = dim_game.merge(team_data_pivot, how='left', on='GAME_NUMBER')


train_data.to_csv(raw_dir / 'features_check2.csv',  sep=',', index=False)

train_data.to_csv(processed_dir / 'training_data.csv',  sep='|', index=False)
# to do:
# add dim_game data
# add other columns to pivot besides these
