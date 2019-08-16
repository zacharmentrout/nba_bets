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


# location of data directory
data_location = '/Users/zach/Documents/git/nba_bets/data'

data_dir = Path(data_location)
raw_dir = data_dir / 'raw'
interim_dir = data_dir / 'interim'
processed_dir = data_dir / 'processed'

# read in data
team_data =  pd.read_csv(raw_dir / 'team_data_raw.csv', sep = '|')

d = team_data


team_stats_per_min = ['AST',
'BLK',
'DREB',
'FG_PCT'
]

#def build_game_level_team_features(d):
team_stats_per_min_list = [calc_stat_per_minute(d, c) for c in team_stats_per_min]

# calculate types of statistics
team_stats_per_min_df = pd.DataFrame(list(zip(*team_stats_per_min_list)),
               columns =[s + '_per_min' for s in team_stats_per_min])
    # AST, BLK

# put it all together
team_stats = pd.concat([team_stats.reset_index(drop=True), team_stats_per_min_df], axis=1)
