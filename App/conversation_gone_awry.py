# In 6

import os

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.feature_selection import f_classif, SelectPercentile

from collections import defaultdict
from functools import partial
from multiprocessing import Pool

from convokit.prompt_types import PromptTypeWrapper
from convokit import PolitenessStrategies
from convokit import Corpus

import matplotlib.pyplot as plt

BASE_DIR = '/'

# In 7

import warnings

warnings.filterwarnings('ignore')

# In 8

AWRY_ROOT_DIR = BASE_DIR + '/conversations-gone-awry-corpus'
awry_corpus = Corpus(AWRY_ROOT_DIR)
awry_corpus.load_info('utterance', ['parsed'])

# In 9

# first, construct a table of conversations that meet the filter criteria (annotation_year = '2018')
kept_conversations = {c.id: c for c in awry_corpus.iter_conversations() if c.meta['annotation_year'] == "2018"}

# next, construct a filtered utterance table containing only the utterances in the filtered conversations
kept_utterances = {}
for convo_id in kept_conversations:
    for utterance in kept_conversations[convo_id].iter_utterances():
        kept_utterances[utterance.id] = utterance

# finally, we overwrite the `conversations` and `utterances` fields of the Corpus object to turn it into a filtered Corpus.
awry_corpus.conversations = kept_conversations
awry_corpus.utterances = kept_utterances

# make sure the size is what we expect
print(len(awry_corpus.conversations))
print(len(awry_corpus.utterances))

# In 10

FULL_ROOT_DIR = BASE_DIR + '/wiki-corpus'

full_corpus = Corpus(FULL_ROOT_DIR)
full_corpus.load_info('utterance', ['parsed'])

# In 11

pt_model = PromptTypeWrapper(n_types=6, use_prompt_motifs=False, root_only=False,
                             questions_only=False, enforce_caps=False, min_support=20, min_df=100,
                             svd__n_components=50, max_dist=1., random_state=1000)
pt_model.fit(full_corpus)

# In 12

for i in range(6):
    print(i)
    pt_model.display_type(i, k=25, corpus=full_corpus)
    print('\n\n')

# In 13

TYPE_NAMES = ['Prompt: Casual', 'Prompt: Moderation', 'Prompt: Coordination', 'Prompt: Contention',
              'Prompt: Editing', 'Prompt: Procedures']

# In 14

awry_corpus = pt_model.transform(awry_corpus)

# In 15

prompt_dist_df = pd.DataFrame(index=awry_corpus.vector_reprs['prompt_types__prompt_dists.6']['keys'],
                              data=awry_corpus.vector_reprs['prompt_types__prompt_dists.6']['vects'])
type_ids = np.argmin(prompt_dist_df.values, axis=1)
mask = np.min(prompt_dist_df.values, axis=1) >= 1.
type_ids[mask] = 6

prompt_dist_df.columns = ['km_%d_dist' % c for c in prompt_dist_df.columns]

# In 16

prompt_dist_df.head()
prompt_type_assignments = np.zeros((len(prompt_dist_df), prompt_dist_df.shape[1] + 1))
prompt_type_assignments[np.arange(len(type_ids)), type_ids] = 1
# noinspection PyTypeChecker
prompt_type_assignment_df = pd.DataFrame(columns=np.arange(prompt_dist_df.shape[1] + 1), index=prompt_dist_df.index,
                                         data=prompt_type_assignments)
prompt_type_assignment_df = prompt_type_assignment_df[prompt_type_assignment_df.columns[:-1]]

# In 17

prompt_type_assignment_df.columns = TYPE_NAMES

# In 18

prompt_type_assignment_df.head()

# In 19

# noinspection PyTypeChecker
ps = PolitenessStrategies(verbose=1000)
awry_corpus = ps.transform(awry_corpus)

# In 20

utterance_ids = awry_corpus.get_utterance_ids()
rows = []
for uid in utterance_ids:
    rows.append(awry_corpus.get_utterance(uid).meta["politeness_strategies"])
politeness_strategies = pd.DataFrame(rows, index=utterance_ids)

# In 21

politeness_strategies.head(10)

# In 22

# first, we need to directly map comment IDs to their conversations. We'll build a DataFrame to do this
comment_ids = []
convo_ids = []
timestamps = []
page_ids = []
for conversation in awry_corpus.iter_conversations():
    for comment in conversation.iter_utterances():
        # section headers are included in the dataset for completeness, but for prediction we need to ignore
        # them as they are not utterances
        if not comment.meta["is_section_header"]:
            comment_ids.append(comment.id)
            convo_ids.append(comment.root)
            timestamps.append(comment.timestamp)
            page_ids.append(conversation.meta["page_id"])
comment_df = pd.DataFrame({"conversation_id": convo_ids, "timestamp": timestamps, "page_id": page_ids},
                          index=comment_ids)

# we'll do our construction using awry conversation ID's as the reference key
awry_convo_ids = set()
# these dicts will then all be keyed by awry ID
good_convo_map = {}
page_id_map = {}
for conversation in awry_corpus.iter_conversations():
    if conversation.meta["conversation_has_personal_attack"] and conversation.id not in awry_convo_ids:
        awry_convo_ids.add(conversation.id)
        good_convo_map[conversation.id] = conversation.meta["pair_id"]
        page_id_map[conversation.id] = conversation.meta["page_id"]
awry_convo_ids = list(awry_convo_ids)
pairs_df = pd.DataFrame({"bad_conversation_id": awry_convo_ids,
                         "conversation_id": [good_convo_map[cid] for cid in awry_convo_ids],
                         "page_id": [page_id_map[cid] for cid in awry_convo_ids]})
# finally, we will augment the pairs dataframe with the IDs of the first and second comment for both
# the bad and good conversation. This will come in handy for constructing the feature matrix.
first_ids = []
second_ids = []
first_ids_bad = []
second_ids_bad = []
for row in pairs_df.itertuples():
    # "first two" is defined in terms of time of posting
    comments_sorted = comment_df[comment_df.conversation_id == row.conversation_id].sort_values(by="timestamp")
    first_ids.append(comments_sorted.iloc[0].name)
    second_ids.append(comments_sorted.iloc[1].name)
    comments_sorted_bad = comment_df[comment_df.conversation_id == row.bad_conversation_id].sort_values(by="timestamp")
    first_ids_bad.append(comments_sorted_bad.iloc[0].name)
    second_ids_bad.append(comments_sorted_bad.iloc[1].name)
pairs_df = pairs_df.assign(first_id=first_ids, second_id=second_ids,
                           bad_first_id=first_ids_bad, bad_second_id=second_ids_bad)


# In 23

def clean_feature_name(feat):
    new_feat = feat.replace('feature_politeness', '').replace('==', '').replace('_', ' ')
    split = new_feat.split()
    first, rest = split[0], ' '.join(split[1:]).lower()
    if first[0].isalpha():
        first = first.title()
    if 'Hashedge' in first:
        return 'Hedge (lexicon)'
    if 'Hedges' in first:
        return 'Hedge (dep. tree)'
    if 'greeting' in feat:
        return 'Greetings'
    cleaner_str = first + ' ' + rest
    #     cleaner_str = cleaner_str.replace('2nd', '2$\mathregular{^{nd}}$').replace('1st', '1$\mathregular{^{st}}$')
    return cleaner_str


# In 24

politeness_strategies_display = politeness_strategies[[col for col in politeness_strategies.columns
                                                       if col not in ['feature_politeness_==HASNEGATIVE==',
                                                                      'feature_politeness_==HASPOSITIVE==']]].copy()
politeness_strategies_display.columns = [clean_feature_name(col) for col in politeness_strategies_display.columns]


# In 25

all_features = politeness_strategies_display.join(prompt_type_assignment_df)


# In 26

tox_first_comment_features =pairs_df[['bad_first_id']].join(all_features, how='left', on='bad_first_id')[all_features.columns]
ntox_first_comment_features =pairs_df[['first_id']].join(all_features, how='left', on='first_id')[all_features.columns]

tox_second_comment_features =pairs_df[['bad_second_id']].join(all_features, how='left', on='bad_second_id')[all_features.columns]
ntox_second_comment_features =pairs_df[['second_id']].join(all_features, how='left', on='second_id')[all_features.columns]


# In 27

def get_p_stars(x):
    if x < .001: return '***'
    elif x < .01: return '**'
    elif x < .05: return '*'
    else: return ''
def compare_tox(df_ntox, df_tox,  min_n=0):
    cols = df_ntox.columns
    num_feats_in_tox = df_tox[cols].sum().astype(int).rename('num_feat_tox')
    num_nfeats_in_tox = (1 - df_tox[cols]).sum().astype(int).rename('num_nfeat_tox')
    num_feats_in_ntox = df_ntox[cols].sum().astype(int).rename('num_feat_ntox')
    num_nfeats_in_ntox = (1 - df_ntox[cols]).sum().astype(int).rename('num_nfeat_ntox')
    prop_tox = df_tox[cols].mean().rename('prop_tox')
    ref_prop_ntox = df_ntox[cols].mean().rename('prop_ntox')
    n_tox = len(df_tox)
    df = pd.concat([
        num_feats_in_tox,
        num_nfeats_in_tox,
        num_feats_in_ntox,
        num_nfeats_in_ntox,
        prop_tox,
        ref_prop_ntox,
    ], axis=1)
    df['num_total'] = df.num_feat_tox + df.num_feat_ntox
    df['log_odds'] = np.log(df.num_feat_tox) - np.log(df.num_nfeat_tox) \
        + np.log(df.num_nfeat_ntox) - np.log(df.num_feat_ntox)
    df['abs_log_odds'] = np.abs(df.log_odds)
    df['binom_p'] = df.apply(lambda x: stats.binom_test(x.num_feat_tox, n_tox, x.prop_ntox), axis=1)
    df = df[df.num_total >= min_n]
    df['p'] = df['binom_p'].apply(lambda x: '%.3f' % x)
    df['pstars'] = df['binom_p'].apply(get_p_stars)
    return df.sort_values('log_odds', ascending=False)


# In 28

first_comparisons = compare_tox(ntox_first_comment_features, tox_first_comment_features)
second_comparisons = compare_tox(ntox_second_comment_features, tox_second_comment_features)


# In 29

# we are now ready to plot these comparisons. the following (rather intimidating) helper function
# produces a nicely-formatted plot:
def draw_figure(ax, first_cmp, second_cmp, title='', prompt_types=6, min_log_odds=.2, min_freq=50, xlim=.85):
    # selecting and sorting the features to plot, given minimum effect sizes and statistical significance
    frequent_feats = first_cmp[first_cmp.num_total >= min_freq].index.union(
        second_cmp[second_cmp.num_total >= min_freq].index)
    lrg_effect_feats = first_cmp[(first_cmp.abs_log_odds >= .2)
                                 & (first_cmp.binom_p < .05)].index.union(second_cmp[(second_cmp.abs_log_odds >= .2)
                                                                                     & (
                                                                                                 second_cmp.binom_p < .05)].index)
    feats_to_include = frequent_feats.intersection(lrg_effect_feats)
    feat_order = sorted(feats_to_include, key=lambda x: first_cmp.loc[x].log_odds, reverse=True)

    # parameters determining the look of the figure
    colors = ['darkorchid', 'seagreen']
    shapes = ['d', 's']
    eps = .02
    star_eps = .035
    xlim = xlim
    min_log = .2
    gap_prop = 2
    label_size = 14
    title_size = 18
    radius = 144
    features = feat_order
    ax.invert_yaxis()
    ax.plot([0, 0], [0, len(features) / gap_prop], color='black')

    # for each figure we plot the point according to effect size in the first and second comment,
    # and add axis labels denoting statistical significance
    yticks = []
    yticklabels = []
    for f_idx, feat in enumerate(features):
        curr_y = (f_idx + .5) / gap_prop
        yticks.append(curr_y)
        try:

            first_p = first_cmp.loc[feat].binom_p
            second_p = second_cmp.loc[feat].binom_p
            if first_cmp.loc[feat].abs_log_odds < min_log:
                first_face = "white"
            elif first_p >= 0.05:
                first_face = 'white'
            else:
                first_face = colors[0]
            if second_cmp.loc[feat].abs_log_odds < min_log:
                second_face = "white"
            elif second_p >= 0.05:
                second_face = 'white'
            else:
                second_face = colors[1]
            ax.plot([-1 * xlim, xlim], [curr_y, curr_y], '--', color='grey', zorder=0, linewidth=.5)

            ax.scatter([first_cmp.loc[feat].log_odds], [curr_y + eps], s=radius, edgecolor=colors[0], marker=shapes[0],
                       zorder=20, facecolors=first_face)
            ax.scatter([second_cmp.loc[feat].log_odds], [curr_y + eps], s=radius, edgecolor=colors[1], marker=shapes[1],
                       zorder=10, facecolors=second_face)

            first_pstr_len = len(get_p_stars(first_p))
            second_pstr_len = len(get_p_stars(second_p))
            p_str = np.array([' '] * 8)
            if first_pstr_len > 0:
                p_str[:first_pstr_len] = '*'
            if second_pstr_len > 0:
                p_str[-second_pstr_len:] = '⁺'

            feat_str = feat + '\n' + ''.join(p_str)
            yticklabels.append(feat_str)
        except Exception as e:
            yticklabels.append('')

    # add the axis labels
    ax.set_xlabel('log-odds ratio', fontsize=14, family='serif')
    ax.set_xticks([-xlim - .05, -.5, 0, .5, xlim])
    ax.set_xticklabels(['on-track', -.5, 0, .5, 'awry'], fontsize=14, family='serif')
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=16, family='serif')
    ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off')
    if title != '':
        ax.text(0, (len(features) + 2.25) / gap_prop, title, fontsize=title_size, family='serif',
                horizontalalignment='center', )
    return feat_order


# In 30

f, ax = plt.subplots(1,1, figsize=(5,10))
_ = draw_figure(ax, first_comparisons, second_comparisons, 'First & second comment')


# In 31

def features_for_convo(convo_id, first_comment_id, second_comment_id):

    # get prompt type features
    try:
        first_prompts = prompt_dist_df.loc[first_comment_id]
    except:
        first_prompts = pd.Series(data=np.ones(len(prompt_dist_df.columns)), index=prompt_dist_df.columns)
    try:
        second_prompts = prompt_dist_df.loc[second_comment_id].rename({c: c + "_second" for c in prompt_dist_df.columns})
    except:
        second_prompts = pd.Series(data=np.ones(len(prompt_dist_df.columns)), index=[c + "_second" for c in prompt_dist_df.columns])
    prompts = first_prompts.append(second_prompts)
    # get politeness strategies features
    first_politeness = politeness_strategies.loc[first_comment_id]
    second_politeness = politeness_strategies.loc[second_comment_id].rename({c: c + "_second" for c in politeness_strategies.columns})
    politeness = first_politeness.append(second_politeness)
    return politeness.append(prompts)


# In 32

convo_ids = np.concatenate((pairs_df.conversation_id.values, pairs_df.bad_conversation_id.values))
feats = [features_for_convo(row.conversation_id, row.first_id, row.second_id) for row in pairs_df.itertuples()] + \
        [features_for_convo(row.bad_conversation_id, row.bad_first_id, row.bad_second_id) for row in pairs_df.itertuples()]
# noinspection PyTypeChecker
feature_table = pd.DataFrame(data=np.vstack([f.values for f in feats]), columns=feats[0].index, index=convo_ids)


# In 33

# in the paper, we dropped the sentiment lexicon based features (HASPOSITIVE and HASNEGATIVE), opting
# to instead use them as a baseline. We do this here as well to be consistent with the paper.
feature_table = feature_table.drop(columns=["feature_politeness_==HASPOSITIVE==",
                                            "feature_politeness_==HASNEGATIVE==",
                                            "feature_politeness_==HASPOSITIVE==_second",
                                            "feature_politeness_==HASNEGATIVE==_second"])


# In 34

feature_table.head(5)


# In 35

def mode(seq):
    vals, counts = np.unique(seq, return_counts=True)
    return vals[np.argmax(counts)]


def run_pred_single(inputs, X, y):
    f_idx, (train_idx, test_idx) = inputs

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    base_clf = Pipeline([("scaler", StandardScaler()), ("featselect", SelectPercentile(f_classif, 10)),
                         ("logreg", LogisticRegression(solver='liblinear'))])
    clf = GridSearchCV(base_clf, {"logreg__C": [10 ** i for i in range(-4, 4)],
                                  "featselect__percentile": list(range(10, 110, 10))}, cv=3)

    clf.fit(X_train, y_train)

    y_scores = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    feature_weights = clf.best_estimator_.named_steps["logreg"].coef_.flatten()
    feature_mask = clf.best_estimator_.named_steps["featselect"].get_support()

    hyperparams = clf.best_params_

    return (y_pred, y_scores, feature_weights, hyperparams, feature_mask)


def run_pred(X, y, fnames, groups):
    feature_weights = {}
    scores = np.asarray([np.nan for i in range(len(y))])
    y_pred = np.zeros(len(y))
    hyperparameters = defaultdict(list)
    splits = list(enumerate(LeaveOneGroupOut().split(X, y, groups)))
    accs = []

    with Pool(os.cpu_count()) as p:
        prediction_results = p.map(partial(run_pred_single, X=X, y=y), splits)

    fselect_pvals_all = []
    for i in range(len(splits)):
        f_idx, (train_idx, test_idx) = splits[i]
        y_pred_i, y_scores_i, weights_i, hyperparams_i, mask_i = prediction_results[i]
        y_pred[test_idx] = y_pred_i
        scores[test_idx] = y_scores_i
        feature_weights[f_idx] = np.asarray([np.nan for _ in range(len(fnames))])
        feature_weights[f_idx][mask_i] = weights_i
        for param in hyperparams_i:
            hyperparameters[param].append(hyperparams_i[param])

    acc = np.mean(y_pred == y)
    pvalue = stats.binom_test(sum(y_pred == y), n=len(y), alternative="greater")

    coef_df = pd.DataFrame(feature_weights, index=fnames)
    coef_df['mean_coef'] = coef_df.apply(np.nanmean, axis=1)
    coef_df['std_coef'] = coef_df.apply(np.nanstd, axis=1)
    return acc, coef_df[['mean_coef', 'std_coef']], scores, pd.DataFrame(hyperparameters), pvalue


def get_labeled_pairs(pairs_df):
    paired_labels = []
    c0s = []
    c1s = []
    page_ids = []
    for i, row in enumerate(pairs_df.itertuples()):
        if i % 2 == 0:
            c0s.append(row.conversation_id)
            c1s.append(row.bad_conversation_id)
        else:
            c0s.append(row.bad_conversation_id)
            c1s.append(row.conversation_id)
        paired_labels.append(i % 2)
        page_ids.append(row.page_id)
    return pd.DataFrame({"c0": c0s, "c1": c1s, "first_convo_toxic": paired_labels, "page_id": page_ids})


def get_feature_subset(labeled_pairs_df, feature_list):
    prompt_type_names = ["km_%d_dist" % i for i in range(6)] + ["km_%d_dist_second" % i for i in range(6)]
    politeness_names = [f for f in feature_table.columns if f not in prompt_type_names]

    features_to_use = []
    if "prompt_types" in feature_list:
        features_to_use += prompt_type_names
    if "politeness_strategies" in feature_list:
        features_to_use += politeness_names

    feature_subset = feature_table[features_to_use]

    c0_feats = feature_subset.loc[labeled_pairs_df.c0].values
    c1_feats = feature_subset.loc[labeled_pairs_df.c1].values

    return c0_feats, c1_feats, features_to_use


def run_pipeline(feature_set):
    print("Running prediction task for feature set", "+".join(feature_set))
    print("Generating labels...")
    labeled_pairs_df = get_labeled_pairs(pairs_df)
    print("Computing paired features...")
    X_c0, X_c1, feature_names = get_feature_subset(labeled_pairs_df, feature_set)
    X = X_c1 - X_c0
    print("Using", X.shape[1], "features")
    y = labeled_pairs_df.first_convo_toxic.values
    print("Running leave-one-page-out prediction...")
    accuracy, coefs, scores, hyperparams, pvalue = run_pred(X, y, feature_names, labeled_pairs_df.page_id)
    print("Accuracy:", accuracy)
    print("p-value: %.4e" % pvalue)
    print("C (mode):", mode(hyperparams.logreg__C))
    print("Percent of features (mode):", mode(hyperparams.featselect__percentile))
    print("Coefficents:")
    print(coefs.sort_values(by="mean_coef"))
    return accuracy


# In 36

feature_combos = [["politeness_strategies"], ["prompt_types"], ["politeness_strategies", "prompt_types"]]
combo_names = []
accs = []
for combo in feature_combos:
    combo_names.append("+".join(combo).replace("_", " "))
    accuracy = run_pipeline(combo)
    accs.append(accuracy)
results_df = pd.DataFrame({"Accuracy": accs}, index=combo_names)
results_df.index.name = "Feature set"


# In 37

# let's see the table
results_df
