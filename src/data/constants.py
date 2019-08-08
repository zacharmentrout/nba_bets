
ROW_IDS_PLAYER = ['GAME_ID', 'TEAM_ID', 'PLAYER_ID']

ROW_IDS_TEAM = ['GAME_ID', 'TEAM_ID']

PLAYER_STATS_TO_GET = {
    BoxScoreAdvancedV2: {
    0: ROW_IDS_PLAYER + ['TEAM_ABBREVIATION', 'TEAM_CITY',
       'PLAYER_NAME', 'START_POSITION', 'COMMENT', 'MIN', 'E_OFF_RATING',
       'OFF_RATING', 'E_DEF_RATING', 'DEF_RATING', 'E_NET_RATING',
       'NET_RATING', 'AST_PCT', 'AST_TOV', 'AST_RATIO', 'OREB_PCT', 'DREB_PCT',
       'REB_PCT', 'TM_TOV_PCT', 'EFG_PCT', 'TS_PCT', 'USG_PCT', 'E_USG_PCT',
       'E_PACE', 'PACE', 'PACE_PER40', 'POSS', 'PIE']
    },
    BoxScoreScoringV2: {
    0: ROW_IDS_PLAYER + ['PCT_FGA_2PT',
       'PCT_FGA_3PT', 'PCT_PTS_2PT', 'PCT_PTS_2PT_MR', 'PCT_PTS_3PT',
       'PCT_PTS_FB', 'PCT_PTS_FT', 'PCT_PTS_OFF_TOV', 'PCT_PTS_PAINT',
       'PCT_AST_2PM', 'PCT_UAST_2PM', 'PCT_AST_3PM', 'PCT_UAST_3PM',
       'PCT_AST_FGM', 'PCT_UAST_FGM']
    },
    BoxScoreTraditionalV2: {
    0: ROW_IDS_PLAYER + ['FGM', 'FGA',
       'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB',
       'DREB', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PF', 'PTS', 'PLUS_MINUS']
    },
    # BoxScoreSummaryV2: [0],
    BoxScoreFourFactorsV2: {
    0: ROW_IDS_PLAYER + ['FTA_RATE', 'OPP_EFG_PCT', 'OPP_FTA_RATE', 'OPP_TOV_PCT', 'OPP_OREB_PCT']
    },
    BoxScoreUsageV2: {
    0: ROW_IDS_PLAYER + ['PCT_FGM',
       'PCT_FGA', 'PCT_FG3M', 'PCT_FG3A', 'PCT_FTM', 'PCT_FTA', 'PCT_OREB',
       'PCT_DREB', 'PCT_REB', 'PCT_AST', 'PCT_TOV', 'PCT_STL', 'PCT_BLK',
       'PCT_BLKA', 'PCT_PF', 'PCT_PFD', 'PCT_PTS']
    },
    BoxScoreMiscV2: {
    0: ROW_IDS_PLAYER + ['PTS_OFF_TOV', 'PTS_2ND_CHANCE', 'PTS_FB', 'PTS_PAINT', 'OPP_PTS_OFF_TOV', 'OPP_PTS_2ND_CHANCE', 'OPP_PTS_FB', 'OPP_PTS_PAINT', 'BLK', 'BLKA', 'PF', 'PFD']
    }
}

TEAM_STATS_TO_GET = {
    BoxScoreAdvancedV2: {
    1: ROW_IDS_TEAM + ['TEAM_CITY',
       'E_OFF_RATING', 'OFF_RATING', 'E_DEF_RATING', 'DEF_RATING',
       'E_NET_RATING', 'NET_RATING', 'AST_PCT', 'AST_TOV', 'AST_RATIO',
       'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'E_TM_TOV_PCT', 'TM_TOV_PCT',
       'EFG_PCT', 'TS_PCT', 'USG_PCT', 'E_USG_PCT', 'E_PACE', 'PACE',
       'PACE_PER40', 'POSS', 'PIE']
    },
    BoxScoreScoringV2: {
    1: ROW_IDS_TEAM + ['PCT_FGA_2PT', 'PCT_FGA_3PT', 'PCT_PTS_2PT', 'PCT_PTS_2PT_MR',
       'PCT_PTS_3PT', 'PCT_PTS_FB', 'PCT_PTS_FT', 'PCT_PTS_OFF_TOV',
       'PCT_PTS_PAINT', 'PCT_AST_2PM', 'PCT_UAST_2PM', 'PCT_AST_3PM',
       'PCT_UAST_3PM', 'PCT_AST_FGM', 'PCT_UAST_FGM']
    },
    BoxScoreTraditionalV2: {
    1: ROW_IDS_TEAM + ['FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA',
       'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PF', 'PTS',
       'PLUS_MINUS'],
    2: ROW_IDS_TEAM + ['STARTERS_BENCH', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PF', 'PTS']
    },
    # BoxScoreSummaryV2: [0],
    BoxScoreFourFactorsV2: {
    1: ROW_IDS_TEAM + ['FTA_RATE', 'OPP_EFG_PCT','OPP_FTA_RATE', 'OPP_TOV_PCT', 'OPP_OREB_PCT']
    },
    BoxScoreUsageV2: {
    1: ROW_IDS_TEAM + ['PCT_FGM', 'PCT_FGA', 'PCT_FG3M', 'PCT_FG3A',
       'PCT_FTM', 'PCT_FTA', 'PCT_OREB', 'PCT_DREB', 'PCT_REB', 'PCT_AST',
       'PCT_TOV', 'PCT_STL', 'PCT_BLK', 'PCT_BLKA', 'PCT_PF', 'PCT_PFD',
       'PCT_PTS']
    },
   BoxScoreMiscV2: {
    1: ROW_IDS_TEAM + ['PTS_OFF_TOV', 'PTS_2ND_CHANCE', 'PTS_FB', 'PTS_PAINT',
       'OPP_PTS_OFF_TOV', 'OPP_PTS_2ND_CHANCE', 'OPP_PTS_FB', 'OPP_PTS_PAINT', 'BLKA', 'PFD']
    },
    LeagueGameFinder: {
    0: ROW_IDS_TEAM + ['SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'GAME_ID',
       'GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'PTS', 'FGM', 'FGA', 'FG_PCT',
       'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB',
       'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS']
    }
}



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
