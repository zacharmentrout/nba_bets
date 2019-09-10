

from nba_api.stats.endpoints.leaguegamefinder import LeagueGameFinder

gf= LeagueGameFinder(league_id_nullable='00',season_type_nullable='Regular Season', season_nullable='2011-12')
gf_df = gf.get_data_frames()[0]
print(gf_df['GAME_ID'].nunique())
