import nba_api
import sys
import pandas as pd
import numpy as np
import requests
from lxml.html import fromstring
from functools import reduce
from operator import itemgetter
from itertools import cycle
from pathlib import Path
import dateutil
import datetime as dt
import traceback
import time
import re
import json
import math


# To do
# Fix get full game sched function to include team abbrev, team name but not home/away id, name abbrev

# from nba_api.stats.static import players
# from nba_api.stats.static import teams
# from nba_api.stats.static import season
from nba_api.stats.endpoints.leaguegamefinder import LeagueGameFinder
from nba_api.stats.endpoints.leaguegamelog import LeagueGameLog

from nba_api.stats.endpoints.boxscoreadvancedv2 import BoxScoreAdvancedV2
from nba_api.stats.endpoints.boxscorescoringv2 import BoxScoreScoringV2
from nba_api.stats.endpoints.boxscoretraditionalv2 import BoxScoreTraditionalV2
from nba_api.stats.endpoints.boxscoresummaryv2 import BoxScoreSummaryV2
from nba_api.stats.endpoints.boxscorefourfactorsv2 import BoxScoreFourFactorsV2
from nba_api.stats.endpoints.boxscoredefensive import BoxScoreDefensive
from nba_api.stats.endpoints.boxscoreusagev2 import BoxScoreUsageV2
from nba_api.stats.endpoints.boxscoremiscv2 import BoxScoreMiscV2
from nba_api.stats.endpoints.teamgamelog import TeamGameLog
from nba_api.stats.endpoints.scoreboardv2 import ScoreboardV2

def odds2prob_ml(odds):
    if math.isnan(odds):
        return(float('nan'))
    if odds < 0:
        return(-odds / (-odds + 100))
    return(100 / (odds + 100))

def convert_odds_american_to_decimal(odds):
    return(1/odds2prob_ml(odds))


def transform_stats_df(stats_df, df_index, stats_obj_class):
    if stats_obj_class == BoxScoreTraditionalV2 and df_index == 2:
        value_cols = set(stats_df.columns) - set(ROW_IDS_TEAM+['STARTERS_BENCH'])
        df_pivot = stats_df.pivot(columns='STARTERS_BENCH',  values=value_cols, index='TEAM_ID')
        old_names = list(df_pivot.columns.values)
        new_names = [str(o[0])+'_'+str(o[1]) for o in old_names]
        df_pivot.columns = new_names
        df_pivot['GAME_ID'] = np.unique(stats_df['GAME_ID'])[0]
        return df_pivot
    else:
        return stats_df

def get_full_game_schedule(season_year, skeleton=False):
    if isinstance(season_year, int):
        season_year = [season_year]
    all_rows = []
    for yr in season_year:
        url_to_get = 'http://data.nba.com/data/10s/v2015/json/mobile_teams/nba/' + str(yr)+'/league/00_full_schedule_week.json'
        req = requests.get(url_to_get)
        sched_data_by_month = req.json()['lscd']
        for month in range(len(sched_data_by_month)):
            sched = sched_data_by_month[month]['mscd']
            for item in sched['g']:
                game_id = item['gid']
                game_date = item['gdte']
                away_team_id = item['v']['tid']
                away_team_ab = item['v']['ta']
                away_team_name  = item['v']['tn']
                home_team_id = item['h']['tid']
                home_team_ab = item['h']['ta']
                home_team_name = item['h']['tn']
                season_id = game_id[2] + str(yr)
                matchup = away_team_ab + ' @ ' + home_team_ab
                new_row = [season_id, game_id, game_date, matchup, home_team_ab, away_team_ab]

                if skeleton:
                    all_rows.append([home_team_id, home_team_ab, home_team_name] + new_row)
                    all_rows.append([away_team_id, away_team_ab, away_team_name] + new_row)
                else:
                    all_rows.append(new_row)
        col_names = ['SEASON_ID', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'TEAM_ABBREVIATION_HOME', 'TEAM_ABBREVIATION_AWAY']
        if skeleton:
            col_names = ['TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME'] + col_names
        game_sched_data = pd.DataFrame(all_rows, columns = col_names)
        game_sched_data['SEASON_TYPE'] = ['Regular Season' if left(s, 1) == '2' else 'Pre Season' for s in game_sched_data['SEASON_ID']]
        game_sched_data.sort_values('GAME_DATE', inplace=True)
    return (game_sched_data)

def get_proxies():
    # Retrieve latest proxies
    proxies_req = Request('https://www.sslproxies.org/')
    proxies_req.add_header('User-Agent', ua.random())
    proxies_doc = urlopen(proxies_req).read().decode('utf8')

    soup = BeautifulSoup(proxies_doc, 'html.parser')
    proxies_table = soup.find(id='proxylisttable')

    # Save proxies in the array
    for row in proxies_table.tbody.find_all('tr'):
        proxies.append({
          'ip':   row.find_all('td')[0].string,
          'port': row.find_all('td')[1].string
        })
    proxies_list = [proxy['ip'] + ':' + proxy['port'] for proxy in proxies]
    return(proxies_list)


def get_boxscore_data(game_id, end_period=0, player_or_team='both',timeout=10, proxy=None):
    bs_adv = BoxScoreAdvancedV2(end_period=end_period,
        game_id=game_id, timeout=timeout, proxy=proxy)
    bs_sco = BoxScoreScoringV2(end_period=end_period,
        game_id=game_id, timeout=timeout, proxy=proxy)
    # bs_tra = BoxScoreTraditionalV2(end_period=end_period,
    #     game_id=game_id)
    # bs_sum = BoxScoreSummaryV2(game_id=game_id)
    bs_fou = BoxScoreFourFactorsV2(end_period=end_period,
        game_id=game_id, timeout=timeout, proxy=proxy)
    # bs_def = BoxScoreDefensive(game_id=game_id)
    bs_usa = BoxScoreUsageV2(end_period=end_period,
        game_id=game_id, timeout=timeout, proxy=proxy)
    bs_mis = BoxScoreMiscV2(end_period=end_period, game_id=game_id, timeout=timeout, proxy=proxy)

    bs_list = [bs_adv, bs_sco, bs_fou, bs_usa, bs_mis] #, bs_sum, bs_fou]

    bs_results_players_df = None
    bs_results_teams_df = None
    if player_or_team == 'player' or player_or_team == 'both':
        bs_results_players_list = [get_boxscore_data_from_data_frames(x, 'player') for x in bs_list]
        bs_results_players_df = reduce(lambda x, y: pd.merge(x, y, on = ROW_IDS_PLAYER), bs_results_players_list)
    if player_or_team == 'team' or player_or_team == 'both':
        bs_results_teams_list = [get_boxscore_data_from_data_frames(x, 'team') for x in bs_list]
        bs_results_teams_df = reduce(lambda x, y: pd.merge(x, y, on = ROW_IDS_TEAM), bs_results_teams_list)

    return {'player': bs_results_players_df, 'team': bs_results_teams_df}

def get_four_factors_data(game_id, end_period=0, player_or_team='both',timeout=10, proxy=None, row_ids_team=['GAME_ID', 'TEAM_ID'], row_ids_player=['GAME_ID', 'TEAM_ID', 'PLAYER_ID']):
    bs_fou = BoxScoreFourFactorsV2(end_period=end_period,
        game_id=game_id, timeout=timeout, proxy=proxy)

    bs_results_players_df = None
    bs_results_teams_df = None

    if player_or_team == 'player' or player_or_team == 'both':
        bs_results_players_df = get_boxscore_data_from_data_frames(bs_fou, 'player')
    if player_or_team == 'team' or player_or_team == 'both':
        bs_results_teams_df = get_boxscore_data_from_data_frames(bs_fou, 'team')

    return {'player': bs_results_players_df, 'team': bs_results_teams_df}



def get_boxscore_data_from_data_frames(stats_obj, player_or_team='player'):
    dfs = stats_obj.get_data_frames()
    if player_or_team == 'player':
        stats_df = dfs[0]
    elif player_or_team == 'team':
        stats_df = dfs[1]
    else:
        raise ValueError('player_or_team argument must be one of {player, team}')

    return stats_df


def get_gamefinder_obj(proxies=None, league_id='00', season_type='Regular Season', season='2015-16'):
    for i in range(20):
        if proxies is None:
            proxy = None
        else:
            proxy = next(proxies)
        try:
            gamefinder = LeagueGameFinder(league_id_nullable=league_id,season_type_nullable=season_type, season_nullable=season, proxy=proxy)
            return(gamefinder)
        except:
            print("Skipping. Connection error")


def get_basic_game_info(proxies=None, league_id='00', season_type='Regular Season', season='2015-16'):
    gamefinder = get_gamefinder_obj(proxies, league_id, season_type, season)
    return(gamefinder.get_data_frames()[0])

def convert_string_to_date(x):
    return([pd.to_datetime(s).date() for s in x])

def left(s, amount):
    return s[:amount]

def right(s, amount):
    return s[-amount:]


def merge_with_prefix_or_suffix(df1, df2, prefix='',suffix='', how='inner', left_on=None, right_on=None, keep_cols=None):
    df_merged = df1.merge(df2, how=how, left_on=left_on, right_on=right_on)
    drop_cols = [s for s in df_merged.columns if '_y' in s]
    if keep_cols is not None:
        drop_cols = list(set(drop_cols) - set([s + '_y' for s in keep_cols]))
    df_merged.drop(drop_cols, inplace=True, axis=1)
    df_merged.columns = [left(s,len(s)-2) if str.endswith(s, '_x') else prefix + left(s,len(s)-2)+suffix if str.endswith(s, '_y') else prefix + s + suffix if s in df2.columns else s for s in df_merged.columns ]
    return(df_merged)


def get_dtypes_dict(dtype_file, col_name_col_name='col_name', dtype_col_name='type'):
    dtypes_get = pd.read_csv(dtype_file,sep='|',dtype='object')
    dtypes_dict = dict(zip(dtypes_get[col_name_col_name], dtypes_get[dtype_col_name]))
    return(dtypes_dict)


def process_raw_team_game_data(raw_data_file, dtype_file, out_dir, out_file_name, dim_directory):

    dtypes_dim_game = get_dtypes_dict(dim_directory / 'dim_game_dtypes.csv')
    dtypes_dim_season = get_dtypes_dict(dim_directory / 'dim_season_dtypes.csv')
    dtypes_dim_matchup = get_dtypes_dict(dim_directory / 'dim_matchup_dtypes.csv')
    dtypes_dim_abbrev = get_dtypes_dict(dim_directory / 'dim_team_alt_abbrev_dtypes.csv')

    dim_game = pd.read_csv(dim_directory / 'dim_game.csv', sep='|', dtype=dtypes_dim_game)
    #dim_team = dim_tables['teams']
    dim_matchup = pd.read_csv(dim_directory / 'dim_matchup.csv', sep='|', dtype=dtypes_dim_matchup)
    dim_abbrev = pd.read_csv(dim_directory / 'dim_team_abbrev.csv', sep='|', dtype=dtypes_dim_abbrev)
    dim_season = pd.read_csv(dim_directory / 'dim_season.csv', sep='|', dtype=dtypes_dim_season)

    # get dtypes
    dtypes_get = pd.read_csv(dtype_file,sep='|',dtype='object')
    dtypes_dict = dict(zip(dtypes_get['col_name'], dtypes_get['type']))

    raw_data = pd.read_csv(raw_data_file, sep='|', dtype=dtypes_dict)
    processed_data = raw_data.copy()
    processed_data = processed_data.merge(dim_season[['SEASON_ID', 'SEASON_NUMBER']], how='left', on='SEASON_ID')

    processed_data = merge_with_prefix_or_suffix(processed_data, dim_abbrev[['TEAM_ABBREVIATION', 'TEAM_NUMBER', 'TEAM_ID']],
        suffix='_HOME',
        how = 'left',
        left_on = 'TEAM_ABBREVIATION_HOME',
        right_on = 'TEAM_ABBREVIATION',
        keep_cols=['TEAM_ID', 'TEAM_NUMBER']
        )

    processed_data = merge_with_prefix_or_suffix(processed_data, dim_abbrev[['TEAM_ABBREVIATION', 'TEAM_NUMBER', 'TEAM_ID']],
        suffix='_AWAY',
        how = 'left',
        left_on = 'TEAM_ABBREVIATION_AWAY',
        right_on = 'TEAM_ABBREVIATION',
        keep_cols=['TEAM_ID', 'TEAM_NUMBER'])

    processed_data = processed_data.merge(dim_abbrev[['TEAM_ABBREVIATION', 'TEAM_NUMBER']],
        how = 'left',
        on = 'TEAM_ABBREVIATION')


    # drop rows that have team_number_away = NaN or team_number_home = NaN
    drop_rows1 = processed_data['TEAM_NUMBER_AWAY'].isna()
    drop_rows2 = processed_data['TEAM_NUMBER_HOME'].isna()
    drop_rows3 = processed_data['TEAM_NUMBER'].isna()
    drop_rows = [x1 or x2 or x3 for x1, x2, x3 in zip(drop_rows1, drop_rows2, drop_rows3)]
    processed_data = processed_data[np.logical_not(drop_rows)]

    cols_to_int = ['TEAM_NUMBER', 'TEAM_NUMBER_HOME', 'TEAM_NUMBER_AWAY']

    processed_data[cols_to_int] = processed_data[cols_to_int].fillna(-1).astype('int64')

    processed_data = processed_data.merge(dim_matchup[['TEAM_NUMBER_HOME', 'TEAM_NUMBER_AWAY', 'MATCHUP_NUMBER', 'PAIRING_NUMBER']], how='left', on=['TEAM_NUMBER_HOME', 'TEAM_NUMBER_AWAY'])

    processed_data['GAME_DATE'] = [dt.datetime.strptime(s, '%Y-%m-%d').date() for s in processed_data['GAME_DATE']]


    processed_data = processed_data.merge(dim_game[['GAME_ID', 'GAME_NUMBER']], how='left', on='GAME_ID')


    processed_data.to_csv(out_dir / out_file_name, sep='|', index=False)

    dtypes = pd.DataFrame({'col_name':processed_data.columns, 'type':processed_data.dtypes})

    dtypes.to_csv(out_dir / (left(out_file_name, len(out_file_name)-4) + '_dtypes.csv'), sep='|')


    return(processed_data)

def generate_dim_tables(raw_data_file, dim_directory, dtype_file):
    ###########
    # seasons
    ###########
    dtypes = get_dtypes_dict(dtype_file)
    team_game_data = pd.read_csv(raw_data_file, sep='|', dtype=dtypes)
    dim_season = team_game_data.groupby(['SEASON_ID', 'SEASON_TYPE']).agg(
        min_GAME_DATE = ('GAME_DATE', 'min'),
        max_GAME_DATE = ('GAME_DATE', 'max'),
        nunique_GAME_ID = ('GAME_ID', 'nunique'),
        ).sort_values('min_GAME_DATE').reset_index(drop=True)

    date_cols = ['min_GAME_DATE', 'max_GAME_DATE']

    dim_season[date_cols] = dim_season[date_cols].apply(pd.to_datetime, errors='raise')

    dim_season[date_cols] = dim_season[date_cols].apply(lambda x: x.dt.date)

    dim_season['START_YEAR'] = [x.year for x in dim_season['min_GAME_DATE']]
    dim_season['END_YEAR'] = [x.year for x in dim_season['max_GAME_DATE']]
    dim_season['DURATION_DAYS'] = [(y - x).days + 1 for x, y in zip(dim_season['min_GAME_DATE'], dim_season['max_GAME_DATE'])]

    dim_season['SEASON_NUMBER'] = np.arange(len(dim_season))
    dim_season['SEASON_NAME'] = [t + ' ' + str(y) for t,y in zip(dim_season['SEASON_TYPE'], dim_season['START_YEAR'])]

    dim_season['NEXT_REGULAR_SEASON_NAME'] = ['Regular Season ' + str(y + 1) if st == 'Regular Season' else 'Regular Season ' + str(y) if st == 'Pre Season' else NULL for st, y in zip(dim_season['SEASON_TYPE'], dim_season['START_YEAR']) ]

    dim_season['NEXT_PRE_SEASON_NAME'] = ['Pre Season ' + str(y + 1) if st == 'Pre Season' else 'Pre Season ' + str(y + 1) if st == 'Regular Season' else NULL for st, y in zip(dim_season['SEASON_TYPE'], dim_season['START_YEAR']) ]

    dim_season['PREVIOUS_REGULAR_SEASON_NAME'] = ['Regular Season ' + str(y - 1) if st == 'Regular Season' else 'Regular Season ' + str(y - 1) if st == 'Pre Season' else NULL for st, y in zip(dim_season['SEASON_TYPE'], dim_season['START_YEAR']) ]

    dim_season['PREVIOUS_PRE_SEASON_NAME'] = ['Pre Season ' + str(y) if st == 'Regular Season' else 'Pre Season ' + str(y - 1) if st == 'Pre Season' else NULL for st, y in zip(dim_season['SEASON_TYPE'], dim_season['START_YEAR']) ]

    dim_season= merge_with_prefix_or_suffix(dim_season, dim_season[['SEASON_NAME', 'SEASON_NUMBER']],
        prefix='NEXT_REGULAR_',
        how = 'left',
        left_on = 'NEXT_REGULAR_SEASON_NAME',
        right_on = 'SEASON_NAME',
        keep_cols = ['SEASON_NUMBER']
        )

    dim_season= merge_with_prefix_or_suffix(dim_season, dim_season[['SEASON_NAME', 'SEASON_NUMBER']],
        prefix='NEXT_PRE_',
        how = 'left',
        left_on = 'NEXT_PRE_SEASON_NAME',
        right_on = 'SEASON_NAME',
        keep_cols = ['SEASON_NUMBER'])

    dim_season= merge_with_prefix_or_suffix(dim_season, dim_season[['SEASON_NAME', 'SEASON_NUMBER']],
        prefix='PREVIOUS_REGULAR_',
        how = 'left',
        left_on = 'PREVIOUS_REGULAR_SEASON_NAME',
        right_on = 'SEASON_NAME',
        keep_cols = ['SEASON_NUMBER'])

    dim_season= merge_with_prefix_or_suffix(dim_season, dim_season[['SEASON_NAME', 'SEASON_NUMBER']],
        prefix='PREVIOUS_PRE_',
        how = 'left',
        left_on = 'PREVIOUS_PRE_SEASON_NAME',
        right_on = 'SEASON_NAME',
        keep_cols = ['SEASON_NUMBER'])


    cols_to_int = ['NEXT_REGULAR_SEASON_NUMBER', 'NEXT_PRE_SEASON_NUMBER', 'PREVIOUS_REGULAR_SEASON_NUMBER', 'PREVIOUS_PRE_SEASON_NUMBER']

    dim_season[cols_to_int] = dim_season[cols_to_int].fillna(-1).astype('int64')

    team_game_data = team_game_data.merge(dim_season[['SEASON_ID', 'SEASON_NUMBER']], how='left', on='SEASON_ID')

    dim_season.to_csv(dim_directory / 'dim_season.csv',  sep='|', index=False)


    dtypes_dim_season = pd.DataFrame({'col_name':dim_season.columns, 'type':dim_season.dtypes})
    dtypes_dim_season.to_csv(dim_directory / 'dim_season_dtypes.csv', sep='|', index=False)


    ###########
    # teams
    ###########
    dim_team = team_game_data[team_game_data['team_recency_rank'] == 1][['TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME']]

    dim_team['TEAM_NUMBER'] = np.arange(len(dim_team))

    dim_team.to_csv(dim_directory / 'dim_team.csv', sep='|', index=False)

    dtypes_dim_team = pd.DataFrame({'col_name':dim_team.columns, 'type':dim_team.dtypes})
    dtypes_dim_team.to_csv(dim_directory / 'dim_team_dtypes.csv', sep='|', index=False)

    ###########
    # team abbreviations
    ###########

    dim_team_alt_abbrev = team_game_data[['TEAM_ID', 'TEAM_ABBREVIATION']].drop_duplicates()

    dim_team_alt_abbrev = merge_with_prefix_or_suffix(dim_team_alt_abbrev, dim_team[['TEAM_ID', 'TEAM_NUMBER']]
        ,how='left'
        ,left_on='TEAM_ID'
        ,right_on='TEAM_ID'
        ,keep_cols = ['TEAM_NUMBER']
        )

    dim_team_alt_abbrev.to_csv(dim_directory / 'dim_team_abbrev.csv', sep='|', index=False)

    dtypes_dim_team_alt_abbrev = pd.DataFrame({'col_name':dim_team_alt_abbrev.columns, 'type':dim_team_alt_abbrev.dtypes})
    dtypes_dim_team_alt_abbrev.to_csv(dim_directory / 'dim_team_alt_abbrev_dtypes.csv', sep='|', index=False)

    # add team information to basic_team_data_table
    team_game_data = merge_with_prefix_or_suffix(team_game_data, dim_team_alt_abbrev[['TEAM_ABBREVIATION', 'TEAM_NUMBER', 'TEAM_ID']],
        suffix='_HOME',
        how = 'left',
        left_on = 'TEAM_ABBREVIATION_HOME',
        right_on = 'TEAM_ABBREVIATION',
        keep_cols=['TEAM_ID', 'TEAM_NUMBER']
        )

    team_game_data = merge_with_prefix_or_suffix(team_game_data, dim_team_alt_abbrev[['TEAM_ABBREVIATION', 'TEAM_NUMBER', 'TEAM_ID']],
        suffix='_AWAY',
        how = 'left',
        left_on = 'TEAM_ABBREVIATION_AWAY',
        right_on = 'TEAM_ABBREVIATION',
        keep_cols=['TEAM_ID', 'TEAM_NUMBER'])

    # drop rows that have team_number_away = NaN or team_number_home = NaN
    drop_rows1 = team_game_data['TEAM_NUMBER_AWAY'].isna()
    drop_rows2 = team_game_data['TEAM_NUMBER_HOME'].isna()
    drop_rows = [x1 or x2 for x1, x2 in zip(drop_rows1, drop_rows2)]
    team_game_data = team_game_data[np.logical_not(drop_rows)]

    cols_to_int = ['TEAM_NUMBER_HOME', 'TEAM_NUMBER_AWAY']

    team_game_data[cols_to_int] = team_game_data[cols_to_int].fillna(-1).astype('int64')

    ###########
    # matchups
    ###########
    dim_matchup = team_game_data.groupby(['TEAM_NUMBER_HOME', 'TEAM_NUMBER_AWAY']).size().reset_index(drop=True).rename(columns={0:'count'}).drop('count',axis=1)

    pairings = pd.DataFrame([[s1, s2] if s1 < s2 else [s2, s1] for s1, s2 in zip(dim_matchup['TEAM_NUMBER_HOME'], dim_matchup['TEAM_NUMBER_AWAY'])], columns = ['TEAM_NUMBER_PAIRING1', 'TEAM_NUMBER_PAIRING2'])

    dim_matchup = pd.concat([dim_matchup, pairings], axis=1)

    dim_matchup['MATCHUP_NUMBER'] = np.arange(len(dim_matchup))


    team_game_data = team_game_data.merge(dim_matchup[['TEAM_NUMBER_HOME', 'TEAM_NUMBER_AWAY', 'MATCHUP_NUMBER']], how='left', on=['TEAM_NUMBER_HOME', 'TEAM_NUMBER_AWAY'])


    # pairing (no home/away distinction)
    dim_pairing = dim_matchup.groupby(['TEAM_NUMBER_PAIRING1', 'TEAM_NUMBER_PAIRING2']).size().reset_index(drop=True).rename(columns={0:'count'}).drop('count',axis=1)

    dim_pairing['PAIRING_NUMBER'] = np.arange(len(dim_pairing))

    dim_matchup = dim_matchup.merge(dim_pairing, how='left', on=['TEAM_NUMBER_PAIRING1', 'TEAM_NUMBER_PAIRING2'])

    dim_matchup.to_csv(dim_directory / 'dim_matchup.csv', sep='|', index=False)

    dtypes_dim_matchup = pd.DataFrame({'col_name':dim_matchup.columns, 'type':dim_matchup.dtypes})
    dtypes_dim_matchup.to_csv(dim_directory / 'dim_matchup_dtypes.csv', sep='|', index=False)

    dim_pairing = merge_with_prefix_or_suffix(dim_pairing, dim_team[['TEAM_NUMBER', 'TEAM_ABBREVIATION']], how='left', left_on='TEAM_NUMBER_PAIRING1', right_on='TEAM_NUMBER', suffix='_PAIRING1')
    dim_pairing = merge_with_prefix_or_suffix(dim_pairing, dim_team[['TEAM_NUMBER', 'TEAM_ABBREVIATION']], how='left', left_on='TEAM_NUMBER_PAIRING2', right_on='TEAM_NUMBER', suffix='_PAIRING2')
    dim_pairing['PAIRING_NAME'] = [s1 + '_vs_' + s2 for s1, s2 in zip(dim_pairing['TEAM_ABBREVIATION_PAIRING1'], dim_pairing['TEAM_ABBREVIATION_PAIRING2'])]


    dim_pairing.to_csv(dim_directory / 'dim_pairing.csv', sep='|', index=False)

    dtypes_dim_pairing= pd.DataFrame({'col_name':dim_pairing.columns, 'type':dim_pairing.dtypes})
    dtypes_dim_pairing.to_csv(dim_directory / 'dim_pairing_dtypes.csv', sep='|', index=False)

    team_game_data = team_game_data.merge(dim_matchup[['MATCHUP_NUMBER', 'PAIRING_NUMBER']], how='left', on='MATCHUP_NUMBER')


    ###########
    # games
    ###########

    # convert game date col to date
    team_game_data['GAME_DATE'] = [dt.datetime.strptime(s, '%Y-%m-%d').date() for s in team_game_data['GAME_DATE']]

    dim_game = team_game_data.groupby(['GAME_ID', 'SEASON_ID', 'GAME_DATE', 'PAIRING_NUMBER', 'MATCHUP_NUMBER', 'TEAM_NUMBER_HOME', 'TEAM_NUMBER_AWAY']).size().reset_index(drop=True).rename(columns={0:'count'}).drop('count',axis=1)

    dim_game = dim_game.merge(dim_season[['SEASON_ID', 'SEASON_NUMBER', 'SEASON_TYPE', 'SEASON_NAME']], how='left', on='SEASON_ID')

    dim_game['GAME_NUMBER'] = np.arange(len(dim_game))

    dim_game.to_csv(dim_directory / 'dim_game.csv', sep='|', index=False)

    dtypes_dim_game = pd.DataFrame({'col_name':dim_game.columns, 'type':dim_game.dtypes})
    dtypes_dim_game.to_csv(dim_directory / 'dim_game_dtypes.csv', sep='|', index=False)

    return_dict = {
    'games':dim_game,
    'teams':dim_team,
    'pairings':dim_pairing,
    'matchups':dim_matchup,
    'team_abbreviations':dim_team_alt_abbrev,
    'seasons':dim_season
    }
    return(return_dict)


def pull_raw_team_game_data(seasons, out_dir, out_file_name, get_upcoming_games=True):

    if not any([s in ALL_AVAILABLE_SEASONS for s in seasons]):
        raise ValueError('Input vector seasons must be in ALL_AVAILABLE_SEASONS')

        # regular season
    team_data_basic_list_regseason = [get_basic_game_info(season=s) for s in seasons]

    # preseason
    team_data_basic_list_preseason = [get_basic_game_info(season=s, season_type = 'Pre Season') for s in seasons]

    team_data_basic_regseason = pd.concat(team_data_basic_list_regseason)
    team_data_basic_preseason = pd.concat(team_data_basic_list_preseason)

    unique_regseason_teams = team_data_basic_regseason['TEAM_ID'].unique()

    # keep only teams from preseason that are in regseason
    keep_preseason_rows = [True if s in unique_regseason_teams else False for s in team_data_basic_preseason['TEAM_ID']]

    team_data_basic_preseason = team_data_basic_preseason[keep_preseason_rows]

    team_data_basic_regseason['SEASON_TYPE'] = 'Regular Season'
    team_data_basic_preseason['SEASON_TYPE'] = 'Pre Season'

    team_data_basic = pd.concat([team_data_basic_regseason, team_data_basic_preseason]).reset_index(drop=True)

    drop_rows = team_data_basic['WL'].isna()
    team_data_basic = team_data_basic[np.logical_not(drop_rows)]

    game_team_count = team_data_basic.groupby('GAME_ID')['TEAM_ID'].count()
    drop_games = set(game_team_count[game_team_count != 2].index)

    drop_rows = [True if s in drop_games else False for s in team_data_basic['GAME_ID']]
    team_data_basic = team_data_basic[np.logical_not(drop_rows)]

    # add other helpful columns
    team_data_basic['team_recency_rank'] = team_data_basic.sort_values(['GAME_DATE'], ascending=[False]).groupby(['TEAM_ID']).cumcount() + 1

    matchup_split1 = pd.DataFrame([re.split(' @ ', s) for s in team_data_basic['MATCHUP']], columns=['TEAM_ABBREVIATION_AWAY1', 'TEAM_ABBREVIATION_HOME1'])

    matchup_split2 = pd.DataFrame([re.split(' vs. ', s) for s in team_data_basic['MATCHUP']], columns=['TEAM_ABBREVIATION_HOME2', 'TEAM_ABBREVIATION_AWAY2'])

    matchup_splits = pd.concat([matchup_split1, matchup_split2], axis=1, ignore_index=False)

    home_away_final = pd.DataFrame([[s1, s2] if s1 is not None and s2 is not None else [s3, s4] for s1, s2, s3, s4 in zip(matchup_splits['TEAM_ABBREVIATION_AWAY1'], matchup_splits['TEAM_ABBREVIATION_HOME1'], matchup_splits['TEAM_ABBREVIATION_AWAY2'], matchup_splits['TEAM_ABBREVIATION_HOME2'])], columns=['TEAM_ABBREVIATION_AWAY', 'TEAM_ABBREVIATION_HOME'])

    team_data_basic = pd.concat([team_data_basic.reset_index(drop=True), home_away_final], axis=1)

    # ff_data = [get_four_factors_data(x, player_or_team='team')['team'] for x in processed_data['GAME_ID'].unique()]

    # ff_data_df = pd.concat(ff_data)


    ##################################
    # get upcoming games
    ##################################
    if get_upcoming_games:
        upcoming_games_year = int(left(UPCOMING_SEASON, 4))
        game_sched_full = get_full_game_schedule(upcoming_games_year, skeleton=True)
        game_ids_new = set(game_sched_full['GAME_ID']) - set(team_data_basic['GAME_ID'])

        keep_rows = [True if s in game_ids_new else False for s in game_sched_full['GAME_ID']]
        game_sched_full = game_sched_full[keep_rows]

        team_data_basic = pd.concat([team_data_basic, game_sched_full], sort=True)

    team_data_basic.to_csv(out_dir / out_file_name, sep='|', index=False)

    dtypes = pd.DataFrame({'col_name':team_data_basic.columns, 'type':team_data_basic.dtypes})

    dtypes.to_csv(out_dir / (left(out_file_name, len(out_file_name)-4) + '_dtypes.csv'), sep='|', index=False)

    return(team_data_basic)


def update_processed_data(base_data_file, new_data_file, dtype_file, merge_index_cols=['TEAM_NUMBER', 'GAME_NUMBER']):

    # get dtypes
    dtypes_get = pd.read_csv(dtype_file,sep='|',dtype='object')
    dtypes_dict = dict(zip(dtypes_get['col_name'], dtypes_get['type']))


    base_data = pd.read_csv(base_data_file, sep='|', dtype=dtypes_dict)
    new_data = pd.read_csv(new_data_file, sep='|', dtype=dtypes_dict)

    base_data.set_index(merge_index_cols, drop=False, inplace=True)
    new_data.set_index(merge_index_cols, drop=False, inplace=True)
    base_data.update(new_data)

    base_data = base_data.astype(dtypes_dict)

    base_data.to_csv(base_data_file, sep='|')


def convert_oddsportal_scrape_output_to_dataframe(odds_json_file):
    with open(odds_json_file, "r") as read_file:
        odds_dict = json.load(read_file)
    seasons = odds_dict['league']['seasons']
    odds_output_list = [pd.DataFrame(s['games']) for s in seasons if len(s['games']) > 0 ]
    odds_output_df = pd.concat(odds_output_list)
    return(odds_output_df)

def process_oddsportal_scrape_output(odds_json_file, dim_team_file, dim_team_dtypes_file, out_dir=None, out_file_name=None):
    odds_df = convert_oddsportal_scrape_output_to_dataframe(odds_json_file)

    cols_to_float = ['odds_home', 'odds_away']

    odds_df[cols_to_float] = odds_df[cols_to_float].astype('float64')

    odds_df['GAME_DATE'] = convert_string_to_date(odds_df['game_datetime'])

    odds_df['odds_dec_home'] = [convert_odds_american_to_decimal(x) for x in odds_df['odds_home']]
    odds_df['odds_dec_away'] = [convert_odds_american_to_decimal(x) for x in odds_df['odds_away']]
    odds_df['implied_prob_home'] = [odds2prob_ml(x) for x in odds_df['odds_home']]
    odds_df['implied_prob_away'] = [odds2prob_ml(x) for x in odds_df['odds_away']]
    odds_df['adj_implied_prob_home'] = [x / (x+y) for x,y in zip(odds_df['implied_prob_home'], odds_df['implied_prob_away'])]
    odds_df['adj_implied_prob_away'] = [y / (x+y) for x,y in zip(odds_df['implied_prob_home'], odds_df['implied_prob_away'])]

    dim_team_dtypes = get_dtypes_dict(dim_team_dtypes_file)
    dim_team = pd.read_csv(dim_team_file, sep='|', dtype=dim_team_dtypes)

    la_clippers_home_rows = odds_df['team_home'] == 'Los Angeles Clippers'
    la_clippers_away_rows = odds_df['team_away'] == 'Los Angeles Clippers'

    odds_df.loc[la_clippers_home_rows, 'team_home'] = 'LA Clippers'
    odds_df.loc[la_clippers_away_rows, 'team_away'] = 'LA Clippers'

    odds_df = merge_with_prefix_or_suffix(odds_df, dim_team[['TEAM_NUMBER', 'TEAM_NAME']], how='left', left_on='team_home', right_on='TEAM_NAME', suffix='_HOME')
    odds_df = merge_with_prefix_or_suffix(odds_df, dim_team[['TEAM_NUMBER', 'TEAM_NAME']], how='left', left_on='team_away', right_on='TEAM_NAME', suffix='_AWAY')

    cols_to_int = ['TEAM_NUMBER_HOME', 'TEAM_NUMBER_AWAY']

    odds_df[cols_to_int] = odds_df[cols_to_int].fillna(-1).astype('int64')

    odds_df.drop_duplicates(subset=['game_datetime', 'TEAM_NUMBER_HOME', 'TEAM_NUMBER_AWAY'], inplace=True)

    if out_dir is not None:
        dtypes = pd.DataFrame({'col_name':odds_df.columns, 'type':odds_df.dtypes})
        dtypes.to_csv(out_dir / (left(out_file_name, len(out_file_name)-4) + '_dtypes.csv'), sep='|', index=False)
        odds_df.to_csv(out_dir / out_file_name, sep='|', index=False)

    return(odds_df)




data_location = '/Users/zach/Documents/git/nba_bets/data'

data_dir = Path(data_location)
raw_dir = data_dir / 'raw'
interim_dir = data_dir / 'interim'
processed_dir = data_dir / 'processed'
external_dir = data_dir / 'external'
dim_dir = processed_dir / 'dim'


base_raw_data_file_name = 'team_game_data_raw.csv'
base_processed_data_file_name = 'team_game_data_processed.csv'
odds_json_file = external_dir / 'nba/NBA.json'

ALL_AVAILABLE_SEASONS = ['2007-08', '2008-09', '2009-10', '2010-11', '2011-12', '2012-13', '2013-14', '2014-15', '2015-16', '2016-17', '2017-18', '2018-19']

UPCOMING_SEASON = '2019-20'

seasons= ALL_AVAILABLE_SEASONS

team_data = pull_raw_team_game_data(seasons, out_dir = raw_dir, out_file_name=base_raw_data_file_name, get_upcoming_games=True)


base_raw_data_file = raw_dir / base_raw_data_file_name

team_data_dtypes = get_dtypes_dict(raw_dir / 'team_game_data_raw_dtypes.csv')
team_data = pd.read_csv(base_raw_data_file, sep='|', dtype=team_data_dtypes)
team_data_dupes = team_data.duplicated(subset=['GAME_ID', 'TEAM_ID'])

dim_tables = generate_dim_tables(raw_data_file=base_raw_data_file, dim_directory=dim_dir, dtype_file=raw_dir / 'team_game_data_raw_dtypes.csv')

dtype_file_raw_team_data = raw_dir / (left(base_raw_data_file_name, len(base_raw_data_file_name)-4) + '_dtypes.csv')

process_raw_team_game_data(raw_data_file=base_raw_data_file, dtype_file=dtype_file_raw_team_data,out_dir=processed_dir, out_file_name=base_processed_data_file_name, dim_directory=dim_dir)

processed_data = pd.read_csv(processed_dir / 'team_game_data_processed.csv', sep='|', dtype= get_dtypes_dict(processed_dir / 'team_game_data_processed_dtypes.csv'))

processed_odds = process_oddsportal_scrape_output(odds_json_file, dim_dir / 'dim_team.csv', dim_dir / 'dim_team_dtypes.csv', out_dir=processed_dir, out_file_name='processed_odds.csv')


