import nba_api
import sys
import pandas as pd
import numpy as np
from functools import reduce
from operator import itemgetter
from nba_api.stats.static import players
from nba_api.stats.static import teams
from nba_api.stats.endpoints.leaguegamefinder import LeagueGameFinder
from nba_api.stats.endpoints.boxscoreadvancedv2 import BoxScoreAdvancedV2
from nba_api.stats.endpoints.boxscorescoringv2 import BoxScoreScoringV2
from nba_api.stats.endpoints.boxscoretraditionalv2 import BoxScoreTraditionalV2
from nba_api.stats.endpoints.boxscoresummaryv2 import BoxScoreSummaryV2
from nba_api.stats.endpoints.boxscorefourfactorsv2 import BoxScoreFourFactorsV2
from nba_api.stats.endpoints.boxscoredefensive import BoxScoreDefensive
from nba_api.stats.endpoints.boxscoreusagev2 import BoxScoreUsageV2
from nba_api.stats.endpoints.boxscoremiscv2 import BoxScoreMiscV2

# get all players
all_players = players.get_players()

# get all teams
all_teams = teams.get_teams()

def get_boxscore_data(game_id, end_period=0, player_or_team='both'):
    bs_adv = BoxScoreAdvancedV2(end_period=end_period,
        game_id=game_id)
    bs_sco = BoxScoreScoringV2(end_period=end_period,
        game_id=game_id)
    # bs_tra = BoxScoreTraditionalV2(end_period=end_period,
    #     game_id=game_id)
    # bs_sum = BoxScoreSummaryV2(game_id=game_id)
    bs_fou = BoxScoreFourFactorsV2(end_period=end_period,
        game_id=game_id)
    # bs_def = BoxScoreDefensive(game_id=game_id)
    bs_usa = BoxScoreUsageV2(end_period=end_period,
        game_id=game_id)
    bs_mis = BoxScoreMiscV2(end_period=end_period, game_id=game_id)

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



# get all games
gamefinder = LeagueGameFinder(league_id_nullable='00', season_type_nullable='Regular Season', season_nullable='2015-16')

games = gamefinder.get_data_frames()[0]

games2 = get_boxscore_data_from_data_frames(gamefinder, player_or_team='team')

game_id = '0021501225'
end_period=0
stats_obj = bs_tra
stats_df = dfs[2][stats_to_get[2]]
df_index = 2
stats_obj_class = type(stats_obj)
player_or_team = 'both'

gid = '0021501225'
res = get_boxscore_data(gid)
res_player = res['player']
res_team = res['team']
g0 = games[games['GAME_ID'] == gid]



team_stats = pd.merge(g0, res_team, on=ROW_IDS_TEAM)
