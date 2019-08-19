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
from nba_api.stats.static import players
from nba_api.stats.static import teams
from nba_api.stats.static import season
from nba_api.stats.endpoints.leaguegamefinder import LeagueGameFinder
from nba_api.stats.endpoints.boxscoreadvancedv2 import BoxScoreAdvancedV2
from nba_api.stats.endpoints.boxscorescoringv2 import BoxScoreScoringV2
from nba_api.stats.endpoints.boxscoretraditionalv2 import BoxScoreTraditionalV2
from nba_api.stats.endpoints.boxscoresummaryv2 import BoxScoreSummaryV2
from nba_api.stats.endpoints.boxscorefourfactorsv2 import BoxScoreFourFactorsV2
from nba_api.stats.endpoints.boxscoredefensive import BoxScoreDefensive
from nba_api.stats.endpoints.boxscoreusagev2 import BoxScoreUsageV2
from nba_api.stats.endpoints.boxscoremiscv2 import BoxScoreMiscV2


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


def get_boxscore_data_from_data_frames(stats_obj, player_or_team='player'):
    dfs = stats_obj.get_data_frames()

    if player_or_team == 'player':
        stats_to_get = PLAYER_STATS_TO_GET[type(stats_obj)]
        row_ids = ROW_IDS_PLAYER
    elif player_or_team == 'team':
        stats_to_get = TEAM_STATS_TO_GET[type(stats_obj)]
        row_ids = ROW_IDS_TEAM
    else:
        raise ValueError('player_or_team argument must be one of {"player", "team"}')

    stats_list = [transform_stats_df(dfs[x][stats_to_get[x]], x, type(stats_obj)) for x in list(stats_to_get.keys())]

    stats_df = reduce(lambda x, y: pd.merge(x, y, on = row_ids), stats_list)

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




# get all games

# proxies = get_proxies()
# proxy_pool = cycle(proxies)

# # get all players
# all_players = players.get_players()


data_location = '/Users/zach/Documents/git/nba_bets/data'

data_dir = Path(data_location)
raw_dir = data_dir / 'raw'
interim_dir = data_dir / 'interim'
processed_dir = data_dir / 'processed'

# all seasons
all_seasons = ['2007-08', '2008-09', '2009-10', '2010-11', '2011-12', '2012-13', '2013-14', '2014-15', '2015-16', '2016-17', '2017-18', '2018-19']

# regular season
team_data_basic_list_regseason = [get_basic_game_info(season=s) for s in all_seasons]

# preseason
team_data_basic_list_preseason = [get_basic_game_info(season=s, season_type = 'Pre Season') for s in all_seasons]

team_data_basic_regseason = pd.concat(team_data_basic_list_regseason)
team_data_basic_preseason = pd.concat(team_data_basic_list_preseason)

# keep only teams from preseason that are in regseason
keep_preseason_rows = [True if s in team_data_basic_regseason['TEAM_ID'].unique() else False for s in team_data_basic_preseason['TEAM_ID']]

team_data_basic_preseason = team_data_basic_preseason[keep_preseason_rows]

team_data_basic_regseason['SEASON_TYPE'] = 'Regular Season'
team_data_basic_preseason['SEASON_TYPE'] = 'Pre Season'

team_data_basic = pd.concat([team_data_basic_regseason, team_data_basic_preseason]).reset_index()

# add other helpful columns
team_data_basic['team_recency_rank'] = team_data_basic.sort_values(['GAME_DATE'], ascending=[False]).groupby(['TEAM_ID']).cumcount() + 1

matchup_split1 = pd.DataFrame([re.split(' @ ', s) for s in team_data_basic['MATCHUP']], columns=['TEAM_ABBREVIATION_AWAY1', 'TEAM_ABBREVIATION_HOME1'])

matchup_split2 = pd.DataFrame([re.split(' vs. ', s) for s in team_data_basic['MATCHUP']], columns=['TEAM_ABBREVIATION_HOME2', 'TEAM_ABBREVIATION_AWAY2'])

matchup_splits = pd.concat([matchup_split1, matchup_split2], axis=1)

home_away_final = pd.DataFrame([[s1, s2] if s1 is not None and s2 is not None else [s3, s4] for s1, s2, s3, s4 in zip(matchup_splits['TEAM_ABBREVIATION_AWAY1'], matchup_splits['TEAM_ABBREVIATION_HOME1'], matchup_splits['TEAM_ABBREVIATION_AWAY2'], matchup_splits['TEAM_ABBREVIATION_HOME2'])], columns=['TEAM_ABBREVIATION_AWAY', 'TEAM_ABBREVIATION_HOME'])

team_data_basic = pd.concat([team_data_basic, home_away_final], axis=1)

team_data_basic.reset_index()


##################################
# make dimension tables
##################################

# seasons
dim_season = team_data_basic.groupby(['SEASON_ID', 'SEASON_TYPE']).agg(
    min_GAME_DATE = ('GAME_DATE', 'min'),
    max_GAME_DATE = ('GAME_DATE', 'max'),
    nunique_GAME_ID = ('GAME_ID', 'nunique'),
    ).sort_values('min_GAME_DATE').reset_index()

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

team_data_basic = team_data_basic.merge(dim_season[['SEASON_ID', 'SEASON_NUMBER']], how='left', on='SEASON_ID')

# dim_season.set_index('SEASON_NUMBER', inplace=True, drop=False)

dim_season.to_csv(raw_dir / 'dim_season.csv',  sep='|', index=False)

# teams
# # get all teams
# all_teams = teams.get_teams()
dim_team = team_data_basic[team_data_basic['team_recency_rank'] == 1][['TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME']]

dim_team['TEAM_NUMBER'] = np.arange(len(dim_team))

dim_team_alt_abbrev = team_data_basic[['TEAM_ID', 'TEAM_ABBREVIATION']].drop_duplicates()
dim_team_alt_abbrev = merge_with_prefix_or_suffix(dim_team_alt_abbrev, dim_team[['TEAM_ID', 'TEAM_NUMBER']]
    ,how='left'
    ,left_on='TEAM_ID'
    ,right_on='TEAM_ID'
    ,keep_cols = ['TEAM_NUMBER']
    )

# add team information to basic_team_data_table
team_data_basic = merge_with_prefix_or_suffix(team_data_basic, dim_team_alt_abbrev[['TEAM_ABBREVIATION', 'TEAM_NUMBER', 'TEAM_ID']],
    suffix='_HOME',
    how = 'left',
    left_on = 'TEAM_ABBREVIATION_HOME',
    right_on = 'TEAM_ABBREVIATION',
    keep_cols=['TEAM_ID', 'TEAM_NUMBER']
    )

team_data_basic = merge_with_prefix_or_suffix(team_data_basic, dim_team[['TEAM_ABBREVIATION', 'TEAM_NUMBER', 'TEAM_ID']],
    suffix='_AWAY',
    how = 'left',
    left_on = 'TEAM_ABBREVIATION_AWAY',
    right_on = 'TEAM_ABBREVIATION',
    keep_cols=['TEAM_ID', 'TEAM_NUMBER'])

# drop rows that have team_number_away = NaN or team_number_home = NaN
drop_rows1 = team_data_basic['TEAM_NUMBER_AWAY'].isna()
drop_rows2 = team_data_basic['TEAM_NUMBER_HOME'].isna()
drop_rows = [x1 or x2 for x1, x2 in zip(drop_rows1, drop_rows2)]
team_data_basic = team_data_basic[np.logical_not(drop_rows)]

cols_to_int = ['TEAM_NUMBER_HOME', 'TEAM_NUMBER_AWAY']

team_data_basic[cols_to_int] = team_data_basic[cols_to_int].fillna(-1).astype('int64')

# matchups
dim_matchup = team_data_basic.groupby(['TEAM_NUMBER_HOME', 'TEAM_NUMBER_AWAY']).size().reset_index().rename(columns={0:'count'}).drop('count',axis=1)

pairings = pd.DataFrame([[s1, s2] if s1 < s2 else [s2, s1] for s1, s2 in zip(dim_matchup['TEAM_NUMBER_HOME'], dim_matchup['TEAM_NUMBER_AWAY'])], columns = ['TEAM_NUMBER_PAIRING1', 'TEAM_NUMBER_PAIRING2'])

dim_matchup = pd.concat([dim_matchup, pairings], axis=1)

dim_matchup['MATCHUP_NUMBER'] = np.arange(len(dim_matchup))

team_data_basic = team_data_basic.merge(dim_matchup[['TEAM_NUMBER_HOME', 'TEAM_NUMBER_AWAY', 'MATCHUP_NUMBER']], how='left', on=['TEAM_NUMBER_HOME', 'TEAM_NUMBER_AWAY'])


# pairing (no home/away distinction)
dim_pairing = dim_matchup.groupby(['TEAM_NUMBER_PAIRING1', 'TEAM_NUMBER_PAIRING2']).size().reset_index().rename(columns={0:'count'}).drop('count',axis=1)

dim_pairing['PAIRING_NUMBER'] = np.arange(len(dim_pairing))

dim_matchup = dim_matchup.merge(dim_pairing, how='left', on=['TEAM_NUMBER_PAIRING1', 'TEAM_NUMBER_PAIRING2'])

dim_pairing = merge_with_prefix_or_suffix(dim_pairing, dim_team[['TEAM_NUMBER', 'TEAM_ABBREVIATION']], how='left', left_on='TEAM_NUMBER_PAIRING1', right_on='TEAM_NUMBER', suffix='_PAIRING1')
dim_pairing = merge_with_prefix_or_suffix(dim_pairing, dim_team[['TEAM_NUMBER', 'TEAM_ABBREVIATION']], how='left', left_on='TEAM_NUMBER_PAIRING2', right_on='TEAM_NUMBER', suffix='_PAIRING2')
dim_pairing['PAIRING_NAME'] = [s1 + '_vs_' + s2 for s1, s2 in zip(dim_pairing['TEAM_ABBREVIATION_PAIRING1'], dim_pairing['TEAM_ABBREVIATION_PAIRING2'])]

team_data_basic = team_data_basic.merge(dim_matchup[['MATCHUP_NUMBER', 'PAIRING_NUMBER']], how='left', on='MATCHUP_NUMBER')

# games
dim_game = team_data_basic.groupby(['GAME_ID', 'SEASON_ID', 'GAME_DATE', 'PAIRING_NUMBER', 'MATCHUP_NUMBER', 'TEAM_NUMBER_HOME', 'TEAM_NUMBER_AWAY']).size().reset_index().rename(columns={0:'count'}).drop('count',axis=1)

dim_game = dim_game.merge(dim_season[['SEASON_ID', 'SEASON_NUMBER', 'SEASON_TYPE', 'SEASON_NAME']], how='left', on='SEASON_ID')

dim_game['GAME_NUMBER'] = np.arange(len(dim_game))

team_data_basic = team_data_basic.merge(dim_game[['GAME_ID', 'GAME_NUMBER']], how='left', on='GAME_ID')

# assign next and previous game numbers
team_data_basic = team_data_basic.merge(dim_team[['TEAM_ID', 'TEAM_NUMBER']], how='left', on='TEAM_ID')

team_data_basic['TEAM_IS_HOME_TEAM'] = [1 if s1 == s2 else 0 for s1, s2 in zip(team_data_basic['TEAM_NUMBER'], team_data_basic['TEAM_NUMBER_HOME'])]

team_data_basic['TEAM_HOME_OR_AWAY'] = ['HOME' if t == 1 else 'AWAY' for t in team_data_basic['TEAM_IS_HOME_TEAM']]

# drop rows with no valid win/loss
drop_rows_wl = team_data_basic['WL'].isna()
team_data_basic = team_data_basic[np.logical_not(drop_rows_wl)]


# sort by team and game date
team_data_basic.sort_values(by=['TEAM_NUMBER',  'GAME_DATE'], inplace=True)


# team_data_basic['games_by_season_cumulative'] = team_data_basic.groupby(['TEAM_NUMBER', 'SEASON_NUMBER']).cumcount()+1

# calculate stats

# write to file
team_data_basic.to_csv(raw_dir / 'game_data_by_team.csv',  sep='|', index=False)

# game_ids = np.unique(team_data['GAME_ID'])

# game_ids_sub = game_ids[0:10]


# bs_mis = BoxScoreMiscV2(end_period=0, game_id=game_ids[0])

# start = time.time()
# data_boxscore = [get_boxscore_data(id) for id in game_ids[0:100]]
# end = time.time()
# print(end - start)

# proxy = next(proxy_pool)
# data_boxscore = get_boxscore_data(game_ids[99], proxy=proxy)


# bs_adv = BoxScoreAdvancedV2(end_period=0,
#         game_id=game_ids[98],  proxy=proxy)


# games2 = get_boxscore_data(game_ids[0])

# games = gamefinder.get_data_frames()[0]

# games2 = get_boxscore_data_from_data_frames(gamefinder, player_or_team='team')

# game_id = '0021501225'
# end_period=0
# stats_obj = bs_tra
# stats_df = dfs[2][stats_to_get[2]]
# df_index = 2
# stats_obj_class = type(stats_obj)
# player_or_team = 'both'

# gid = '0021501225'
# res = get_boxscore_data(gid)
# res_player = res['player']
# res_team = res['team']
# g0 = games[games['GAME_ID'] == gid]



# team_stats = pd.merge(g0, res_team, on=ROW_IDS_TEAM)
