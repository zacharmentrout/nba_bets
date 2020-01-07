#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 2010

@author: zach
"""
import os
import re

os.chdir('/Users/zach/Documents/git/nba_bets/src/models/')

#import importlib.util
from train_model_functions import *

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
'GAME_DATE_week_number',
'TEAM_FEATURE_cumulative_win_pct_COURT_AWAY',
'TEAM_FEATURE_cumulative_win_pct_COURT_HOME',
'TEAM_FEATURE_cumulative_pt_pct_COURT_AWAY',
'TEAM_FEATURE_cumulative_pt_pct_COURT_HOME',
'TEAM_FEATURE_OFF_RATING_ewma_pythag_HOME_adj',
'TEAM_FEATURE_OFF_RATING_ewma_pythag_AWAY_adj',
'TEAM_FEATURE_cumulative_pt_pct_COURT_HOME_AWAY_diff',
#'TEAM_FEATURE_cumulative_pt_pct_COURT_HOME_AWAY_frac',
'TEAM_FEATURE_cumulative_win_pct_COURT_HOME_AWAY_diff',
]

predictors = [
# 'TEAM_FEATURE_cumulative_win_pct_COURT_AWAY',
# 'TEAM_FEATURE_cumulative_win_pct_COURT_HOME',
# 'TEAM_FEATURE_cumulative_pt_pct_COURT_AWAY',
# 'TEAM_FEATURE_cumulative_pt_pct_COURT_HOME',
# 'TEAM_FEATURE_OFF_RATING_ewma_pythag_HOME_adj',
# 'TEAM_FEATURE_OFF_RATING_ewma_pythag_AWAY_adj',
'TEAM_FEATURE_cumulative_pt_pct_COURT_HOME_AWAY_diff',
#'TEAM_FEATURE_cumulative_pt_pct_COURT_HOME_AWAY_frac',
'TEAM_FEATURE_cumulative_win_pct_COURT_HOME_AWAY_diff',
#'TEAM_FEATURE_cumulative_win_pct_COURT_HOME_AWAY_frac',
#'TEAM_FEATURE_cumulative_pt_pct_COURT_pythag_HOME',
#'TEAM_FEATURE_cumulative_win_pct_COURT_pythag_HOME'
#'implied_prob_home',
#'implied_prob_away',
# 'TEAM_FEATURE_OFF_RATING_ewma_pythag_adj_HOME_AWAY_diff',
#'TEAM_FEATURE_cumulative_count_GAME_NUMBER_AWAY',
 'TEAM_FEATURE_cumulative_count_GAME_NUMBER_HOME',
#'TEAM_FEATURE_pt_pct_prev_HOME',
# 'TEAM_FEATURE_pt_pct_prev_AWAY',
#'TEAM_FEATURE_pt_pct_prev_COURT_HOME',
# 'TEAM_FEATURE_pt_pct_prev_COURT_AWAY',
#'TEAM_FEATURE_pct_pct_prev_HOME_AWAY_diff',
'TEAM_FEATURE_pct_pct_prev_COURT_HOME_AWAY_diff',
#'SEASON_NUMBER'
# 'implied_prob_home',
# 'implied_prob_away',
# 'amax_odds_dec_home',
# 'amax_odds_dec_away'
]

model_param_dict = {
    'learning_rate':[0.001],
    'n_estimators':[750],
    #'max_depth':[3,5,10],
    'min_samples_leaf':[1],
    'min_child_weight':[1],
    'gamma':[0.5],
    'subsample':[1],
    'colsample_bytree':[1.0],
    'objective':['binary:logistic'],
    'nthread':[1],
    'scale_pos_weight':[1],
    'seed':[223],
    'train_frac':[0.5],
    'bet_dist_type':['max-ep', 'opt'],
    'bet_budget':[100],
    'min_exp_roi':[0.1],
    #'conf_threshold':[0.2],

    # functions
     'training_prediction_functions':[(train_rf_classifier, predict_with_rf_classifier)],
    #'training_prediction_functions':[(train_logistic_cv_classifier, predict_with_logistic_classifier)],
    'model_performance_function':[calc_model_performance],
    'bet_recommendation_function':[calc_bet_distribution],
    'bet_performance_function':[calc_total_profit],
    'modeling_round_var':[['SEASON_NUMBER']],
    #'modeling_round_var':[['GAME_DATE_year','GAME_DATE_week_number']],
    'betting_round_params':[(['GAME_DATE_year','GAME_DATE_week_number'], [2016, 46], [2017, 15])]
    ,'penalty':['elasticnet']
    ,'max_iter':[10000]

    #,'C':[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1]
}



##################
# get model parameters
##################

model_params = make_parameter_dict_list(model_param_dict)

##################
# read in data
##################

data_dir = Path(data_location)
processed_dir = data_dir / 'processed'
raw_dir = data_dir / 'raw'

train_data_init = pd.read_csv(processed_dir / 'training_data.csv', in_delim)

# train_data_init = train_data_init[~np.isnan(train_data_init['implied_prob_home'])]


# ##################
# # test model training
# ##################

simulator = Simulator(param_dict_list=model_params, training_data=train_data_init, predictors=predictors, target=target)

# todo use function names in model param dict and grab functions later in constructor with locals()['function_name']
# fails when game_date week number is modeling round variable(s)
simulator.simulate()
sim_out = simulator.simulation_output()
agg_sim_out = simulator.aggregate_simulation_output()

sim_out.sort_values(['GAME_DATE_year','GAME_DATE_week_number'])


sim_out.sort_values(['GAME_DATE_year','GAME_DATE_week_number'])
sim_out['year_week'] = [str(s1)+'-'+str(s2) for s1, s2 in zip(sim_out['GAME_DATE_year'],sim_out['GAME_DATE_week_number'])]

sim_out['cumulative_bet_amount'] = sim_out[['param_num','total_bet_amount']].groupby(['param_num']).cumsum()

sim_out['cumulative_roi'] = sim_out['cumulative_profit'] / sim_out['cumulative_bet_amount']

sim1 = sim_out[sim_out['param_num'] == 1]


x = sim1['year_week']# np.linspace(1,sim1.shape[0],sim1.shape[0])
fig, ax = plt.subplots(2,1)
#ax[0].set(title="Cumulative profit by week", xlabel="Year-Week", ylabel="Cumulative Profit")
#ax[0].set_xticklabels(sim1['year_week'],rotation=45,ha='right')

ax[0].set(title="Cumulative ROI by week", xlabel="Year-Week", ylabel="Cumulative ROI")
ax[0].set_xticklabels(sim1['year_week'],rotation=45,ha='right')
ax[1].set(title="Log loss by week", xlabel="Year-Week", ylabel="Log Loss")
ax[1].set_xticklabels(sim1['year_week'],rotation=45,ha='right')


for k in sim_out['param_num'].unique():
    #y0 = sim_out[sim_out['param_num'] == k]['cumulative_profit']
    #ax[0].plot(x,y0,label='param num '+str(k))
    y1 = sim_out[sim_out['param_num'] == k]['roi']
    ax[0].plot(x,y1,label='param num '+str(k))
    y2 = sim_out[sim_out['param_num'] == k]['logloss']
    ax[1].plot(x,y2,label='param num '+str(k))

plt.tight_layout()
plt.legend()
plt.show()



agg_sim_out_grouped = agg_sim_out[['param_num_agg', 'total_profit', 'total_roi', 'total_exp_profit','total_bets_placed', 'total_correct_bets', 'total_games','total_available_bets']].groupby(['param_num_agg'])

agg_sim_out2 = agg_sim_out_grouped.agg([np.size
                             ,  np.mean
                             ,  np.std
                             ,  np.min
                             ,  quartile1
                             ,  np.median
                             ,  quartile3
                             ,  np.max])

names_old = list(agg_sim_out2.columns.values)
new_names = [str(o[1])+'_'+str(o[0]) for o in names_old]
agg_sim_out2.columns = new_names

agg_sim_out2.sort_values(['amin_total_roi'], ascending=False)

mod = simulator.simulations[0].recommender.modeler.model_object.base_estimator

feature_importances = pd.DataFrame(mod.feature_importances_,
                                   index = predictors,
                                    columns=['importance']).sort_values('importance',                                                                 ascending=False)

labels = list(feature_importances.index)
x = np.arange(len(labels))  # the label locations
y = feature_importances['importance']
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, y, width, label='Men')
plt.tight_layout()

ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation='vertical')
ax.legend()
plt.show()

plot_xgb_feat_importance(mod, predictors, 'gain', 'blue', 30)



model_param_dict = {
    'learning_rate':[0.001],
    'n_estimators':[50],
    'max_depth':[5],
    'min_child_weight':[1],
    'gamma':[0.5],
    'subsample':[1],
    'colsample_bytree':[1.0],
    'objective':['binary:logistic'],
    'nthread':[1],
    'scale_pos_weight':[1],
    'seed':[4445],
    'train_frac':[0.8],
    'bet_dist_type':['opt', 'max-ep'],
    'bet_budget':[100],
    'min_exp_roi':[0.1],
    #'conf_threshold':[0.0]

    # functions
    # 'training_prediction_functions':[(train_xgb_classifier, predict_with_xgb_classifier)],
    'training_prediction_functions':[(train_logistic_classifier, predict_with_logistic_classifier)],
    'model_performance_function':[calc_model_performance],
    'bet_recommendation_function':[calc_bet_distribution],
    'bet_performance_function':[calc_total_profit],
    'modeling_round_var':[['SEASON_NUMBER']],
    #'modeling_round_var':[['GAME_DATE_year','GAME_DATE_week_number']],
    'betting_round_params':[(['GAME_DATE_year','GAME_DATE_week_number'], [2016, 46], [2018, 15])]
    #,'penalty':['l1', 'l2']
    #,'max_iter':[1000]
    #,'C':[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1]
}


# Test models
train1 = train_data_init[train_data_init['SEASON_NUMBER']<=21].copy()
test1 = train_data_init[train_data_init['SEASON_NUMBER']==23].copy()


train1.sort_values('GAME_DATE')


params_xgb = {
    'learning_rate':0.0001,
    'min_samples_leaf':1,
    'n_estimators':750,
    #'max_depth':5,
    'min_child_weight':1,
    'gamma':0.5,
    'subsample':1,
    'colsample_bytree':1.0,
    'objective':'binary:logistic',
    'nthread':1,
    'scale_pos_weight':1,
    'seed':445,
    'train_frac':0.5,
    'penalty':'elasticnet'
    ,'max_iter':10000
    #,'C':[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1]
}


mod_xgb = train_xgb_classifier(train1, predictors, target, params=params_xgb)
mod_rf = train_rf_classifier(train1, predictors,target,params=params_xgb)
mod_logit = train_logistic_cv_classifier(train1, predictors, target, params=params_xgb)



pred_xgb = predict_with_xgb_classifier(mod_xgb, test1, predictors)
pred_rf = predict_with_rf_classifier(mod_rf,test1, predictors)
pred_logit = predict_with_logistic_classifier(mod_logit, test1, predictors)
pred_avgodds = predict_with_avg_odds(None, test1, None)
pred_naive = predict_with_naive_classifier(None, test1, None)


plot_reliability_curve(y, pre)


plot_xgb_feat_importance(mod_xgb.base_estimator, predictors)


ll_xgb = metrics.log_loss(test1['WIN_HOME'], pred_xgb)
ll_rf = metrics.log_loss(test1['WIN_HOME'],pred_rf)
ll_logit = metrics.log_loss(test1['WIN_HOME'], pred_logit)
ll_avgodds = metrics.log_loss(test1[~np.isnan(test1['implied_prob_home'])] ['WIN_HOME'], pred_avgodds[~np.isnan(pred_avgodds)])
ll_naive = metrics.log_loss(test1[~np.isnan(test1['amax_odds_dec_home'])] ['WIN_HOME'], pred_naive[~np.isnan(pred_naive)])

bs_rf = metrics.brier_score_loss(test1['WIN_HOME'],pred_rf)
bs_xgb = metrics.brier_score_loss(test1['WIN_HOME'],pred_xgb)
bs_naive = metrics.brier_score_loss(test1[~np.isnan(test1['amax_odds_dec_home'])] ['WIN_HOME'], pred_naive[~np.isnan(pred_naive)])
plot_reliability_curve(test1['WIN_HOME'],pred_xgb)
plot_reliability_curve(test1['WIN_HOME'],pred_rf)
plot_reliability_curve(test1['WIN_HOME'],pred_logit)
plot_reliability_curve(test1['WIN_HOME'],pred_avgodds)
plot_reliability_curve(test1['WIN_HOME'],pred_naive)

acc_xgb = compare_classifier_accuracy(test1['WIN_HOME'], pred_xgb, pred_naive)
acc_rf = compare_classifier_accuracy(test1['WIN_HOME'], pred_rf, pred_naive)
acc_logit = compare_classifier_accuracy(test1['WIN_HOME'], pred_logit, pred_naive)
acc_avgodds = compare_classifier_accuracy(test1['WIN_HOME'], pred_avgodds, pred_naive)

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


# mod =  train_logistic_classifier(train1, predictors, target, model_params[0])

# mod.fit(train1[predictors], train1[target])

# pred = mod.predict_proba(test1[predictors])

# mod = LogisticRegression(model_params[0])



