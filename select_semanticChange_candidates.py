from scipy.misc import derivative
import pandas as pd
import numpy as np
import argparse
import scipy
import ujson
import math
import sys
import re

from lmfit.models import LognormalModel, StepModel

parser = argparse.ArgumentParser()
parser.add_argument("--occByMonth", type=str,
                    help="path to the dictionary containing the number of occurrences by month of the words",
                    required=True)
parser.add_argument("--occ", type=str, help="path to the dictionary containing the number of occurrences of the words",
                    required=True)
parser.add_argument("--embMonth", type=str,
                    help="path to the directory containing the average monthly embeddings by word", required=True)
parser.add_argument("--out", type=str, help="path to the output directory", required=True)
parser.add_argument("--pathID", type=str, help="path to the dictionary containing the ID of words", required=True)

args = parser.parse_args()

path_occByMonth = args.occByMonth
path_occ = args.occ
path_idForm = args.pathID
path_embMonth = args.embMonth
path_out = args.out

idByWord = ujson.load(open(path_idForm))
idByWord_inv = {str(v): k for k, v in idByWord.items()}
occ = ujson.load(open(path_occ))
occ_byMonth = ujson.load(open(path_occByMonth))
months = sorted([l.rstrip() for l in open(path_embMonth + "months_ordered.nfo")])

wordsToKeep = set([k for k,v in occ.items() if (v>10000 and v<200000) and (re.match("^[\w'-]+$", idByWord_inv[k])) and (not re.match("^\d+$", idByWord_inv[k]))])
wordsToKeep_2 = [w for w in wordsToKeep if
                 all(w in occ_byMonth[m] and occ_byMonth[m][w] >= 50 for m in months)]

months = [line.rstrip() for line in open(path_embMonth + "months_ordered.nfo")]
# just 2014-01 -> 2018-12
months = months[22:-2]

# For each word, the cosine distance between the current month and the first month (2014-01) is retrieved for each month.
dic_cos = {}

for k, idWord in enumerate(wordsToKeep_2):

    if k%100==0 :
        sys.stdout.write("\r"+str(k+1))

    word = idByWord_inv[idWord]

    dic_cos[word] = {"id": idWord, "dist": []}

    embeddings_month = np.load(path_embMonth + str(idWord) + "_byMonth_emb.npz", allow_pickle=True)["arr_0"]
    embeddings_month = embeddings_month[22:-2]

    dic_months = {}
    for i, month in enumerate(months):
        dic_months[month] = embeddings_month[i]

    for i, m in enumerate(dic_months):
        dic_cos[word]["dist"].append(scipy.spatial.distance.cosine(dic_months[months[0]], dic_months[m]))

dists = {word: dic_cos[word]["dist"] for word in dic_cos}
df_5years = pd.DataFrame.from_dict(dists, orient="index")

# Relative distances are recovered to ensure that the distances of all words are comparable.
df_5years_rel = df_5years.iloc[:, :].copy()
for i, form in enumerate(df_5years.index):
    for col in df_5years_rel.columns:
        df_5years_rel.loc[form, col] = df_5years.loc[form, col] / df_5years.iloc[i, :].sum()

# Saving data
df_5years.to_csv(path_out + "df_5years_dist.csv")
df_5years_rel.to_csv(path_out + "df_5years_dist_rel.csv")

"""
 ----------------------------------------------------------------------
| fitting of distance curves to select candidates for semantic changes |
 ----------------------------------------------------------------------
"""

# curve fitting function from a reference function (logNormal or logistic)
logNormal_model = LognormalModel()
logistic_model = StepModel(form='logistic')


def fitting(form, log=False):
    x = [i for i in range(60)]
    y = df_5years_rel.loc[form].rolling(window=3, min_periods=0).mean()

    if log:

        params = logistic_model.guess(y, x=x)
        model = logistic_model

    else:
        params = logNormal_model.guess(y, x=x)
        params.add("sigma", value=2)
        model = logNormal_model

    result = model.fit(y, params, x=x)

    if log:
        return {'form': form,
                'sigma': result.params['sigma'].value,
                'sigma_err': result.params['sigma'].stderr,
                'center': result.params['center'].value,
                'center_err': result.params['center'].stderr,
                'amplitude': result.params['amplitude'].value,
                'amplitude_er': result.params['amplitude'].stderr,
                'redchi': result.redchi,
                'chisqr': result.chisqr}

    else:
        maxPoint = np.where(result.best_fit == np.max(result.best_fit))[0][0]
        return {'form': form,
                'sigma': result.params['sigma'].value,
                'sigma_err': result.params['sigma'].stderr,
                'center': result.params['center'].value,
                'center_err': result.params['center'].stderr,
                'amplitude': result.params['amplitude'].value,
                'amplitude_er': result.params['amplitude'].stderr,
                'height': result.params['height'].value,
                'height_err': result.params['height'].stderr,
                'fwhm': result.params['fwhm'].value,
                'fwhm_err': result.params['fwhm'].stderr,
                'redchi': result.redchi,
                'chisqr': result.chisqr,
                'maxPoint': maxPoint}


# function that return the diffusion phases of a word
def phases_delimitation(form, log=False):
    x = [i for i in range(60)]
    y = df_5years_rel.loc[form].rolling(window=3, min_periods=0).mean()

    if log:

        params = logistic_model.guess(y, x=x)
        model = logistic_model

    else:
        params = logNormal_model.guess(y, x=x)
        params.add("sigma", value=2)
        model = logNormal_model

    result = model.fit(y, params, x=x)

    # detection of diffusion phases
    def f(x):
        ampl = result.params['amplitude'].value
        sigma = result.params['sigma'].value
        center = result.params['center'].value
        if log:
            return ampl * (1 - (1 / (1 + math.exp((x - center) / sigma))))
        else:
            return (ampl / (sigma * math.sqrt(2 * math.pi))) * (
                        (math.exp(-((math.log(x) - center) ** 2 / (2 * sigma ** 2)))) / x)

    maxPoint = np.where(result.best_fit == np.max(result.best_fit))[0][0]

    values_deriv_3 = []
    if log:
        for x2 in range(0, 60):
            values_deriv_3.append(derivative(f, x2, n=3, order=5, dx=1))
    else:
        values_deriv_3 = [0]
        for x2 in range(1, 60):
            values_deriv_3.append(derivative(f, x2, n=3, order=5, dx=0.1))

    periods = {
        "innovation": (min(x), values_deriv_3.index(max(values_deriv_3[:values_deriv_3.index(min(values_deriv_3))]))),
        "propagation": (values_deriv_3.index(max(values_deriv_3[:values_deriv_3.index(min(values_deriv_3))])),
                        values_deriv_3.index(max(values_deriv_3[values_deriv_3.index(min(values_deriv_3)):]))),
        "fixation": (values_deriv_3.index(max(values_deriv_3[values_deriv_3.index(min(values_deriv_3)):])), max(x))}

    return (result.best_fit, periods)


lmfit_results = []
for i, word in enumerate(df_5years.index.tolist()):
    sys.stdout.write("\r"+str(i+1))
    lmfit_results.append(fitting(word, log=True))
df_log = pd.DataFrame.from_records(lmfit_results, index="form")

lmfit_results = []
for i, word in enumerate(df_5years.index.tolist()):
    lmfit_results.append(fitting(word, log=False))
df_logNorm = pd.DataFrame.from_records(lmfit_results, index="form")

# Saving data
df_log.to_csv(path_out + "lmfit_log_results_dist.csv")
df_logNorm.to_csv(path_out + "lmfit_logNorm_results_dist.csv")

# Selection with parameters of the curve fitting results
logNorm_select = df_logNorm[(df_logNorm.fwhm >= 4) & (df_logNorm.fwhm <= 40) & (df_logNorm.redchi <= 0.00005) & (
            df_logNorm.amplitude <= 1.1) & (df_logNorm.maxPoint >= 21) & (df_logNorm.maxPoint <= 46) & (
                                        ((df_logNorm.center <= 3.6) & (df_logNorm.sigma <= 0.65)) | (
                                            (df_logNorm.center > 3.6) & (df_logNorm.center <= 3.8) & (
                                                df_logNorm.sigma <= 0.35)) | (
                                                    (df_logNorm.center > 3.8) & (df_logNorm.sigma <= 0.15)))]
logistic_select = df_log[(((df_log.center >= 16) & (df_log.center <= 31) & (df_log.sigma <= 8)) | (
            (df_log.center > 31) & (df_log.center <= 46) & (df_log.sigma <= 7))) & (df_log.redchi < 0.00005) & (
                                     df_log.amplitude > 0.02) & (df_log.center_err < 5)]

buzzes = set(logNorm_select.index.tolist())
changes = set(logistic_select.index.tolist())

dic = {}
for form in changes:
    infos = phases_delimitation(form, log=True)
    best_fit = infos[0]
    periods = infos[1]
    prop_beg = periods["propagation"][0]
    prop_end = periods["propagation"][1]
    # Remove those that have no innovation phase and are also categorised as buzz with a higher redchi.
    if prop_beg!=0 and form not in ["never", "marins"] :
        dic[idByWord[form]] = {"best_fit":list(best_fit), "prop_beg":prop_beg, "prop_end":prop_end, "type":"change", "word":form}

for form in buzzes:
    infos = phases_delimitation(form, log=False)
    best_fit = infos[0]
    periods = infos[1]
    prop_beg = periods["propagation"][0]
    prop_end = periods["propagation"][1]
    dic[idByWord[form]] = {"best_fit": list(best_fit), "prop_beg": prop_beg, "prop_end": prop_end, "type":"buzz", "word":form}

ujson.dump(dic, open(path_out+"candidates.json", "w"))