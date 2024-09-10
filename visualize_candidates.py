from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
from multiprocessing import Pool
from datetime import datetime
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import logging
import scipy
import ujson
import umap
import re
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('visualize_candidates.log')
handler.setFormatter(logging.Formatter("%(asctime)s; %(levelname)s; %(message)s"))
logger.addHandler(handler)

parser = argparse.ArgumentParser()
parser.add_argument("--df", type=str, help="path to the dataframe containing the (relative) cosine distances of each mont to the first month", required=True)
parser.add_argument("--pathID", type=str, help="path to the dictionary containing the ID of words", required=True)
parser.add_argument("--dic_infos", type=str, help="path to the dictionary containing all the selected candidates, identified by their ID, with its type, the values of the fitted trajectory, the start and end of its propagation period, and the corresponding word", required=True)
parser.add_argument("--path_in", type=str, help="path to the directory containing embedding npz files of words, label files, and tweets files", required=True)
parser.add_argument("--output_data", type=str, help="name of the output directory for graphics", required=True)
parser.add_argument("--output_html", type=str, help="name of the output directory for html files", required=True)
parser.add_argument("--output_pdf", type=str, help="name of the output directory pdf files", required=True)

args = parser.parse_args()

logger.info("Charging data.")
df_5years_rel = pd.read_csv(args.df, index_col=0)
idByWord = ujson.load(open(args.pathID))
idByWord_inv = {str(v):k for k,v in idByWord.items()}
dic_infos = ujson.load(open(args.dic_infos))
path_in = args.path_in
output_data = args.output_data
output_html = args.output_html
output_pdf = args.output_pdf

logger.info("--df : "+args.df+", --pathID : "+args.pathID+", --dic_infos : "+args.dic_infos+", --path_in : "+args.path_in+", --output_data : "+args.output_data+", --output_html : "+args.output_html+", --output_pdf : "+args.output_pdf)

# we retrieve months to treat
start = datetime(2014, 1, 1)
end = datetime(2018, 12, 1)
months = []
current_date = start
while current_date <= end:
    months.append(current_date.strftime("%Y-%m"))
    current_date += relativedelta(months=1)
logger.info("Charging data - ended.")


def viewClusters(allEmb, idWord, labels):

    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_result = reducer.fit_transform(allEmb)
    df_umap = pd.DataFrame(umap_result, columns=["dim_1", "dim_2"])
    df_umap["label"] = labels

    plt.figure(figsize=[4, 4])
    sns.scatterplot(data=df_umap, x="dim_1", y="dim_2", hue="label", palette=sns.color_palette(), s=5).set(
        title='UMAP projection')
    plt.tight_layout()
    plt.savefig(output_data+idWord + "_clusterScatter.png")

    return df_umap

def fit(allEmb, idWord, labels) :
    word = idByWord_inv[idWord]
    plt.figure(figsize=[12,4])
    plt.title("Ajustement de courbe", fontsize=15)
    plt.plot(df_5years_rel.loc[word].values, color="#084C61", linewidth=4, label="Distance au premier mois")
    plt.plot(dic_infos[idWord]["best_fit"], color="#E3B505", linewidth=4, label="Meilleur ajustement")
    plt.axvspan(0, dic_infos[idWord]["prop_beg"], color="grey", alpha=0.1)
    plt.axvspan(dic_infos[idWord]["prop_beg"], dic_infos[idWord]["prop_end"], color="grey", alpha=0.3)
    plt.axvspan(dic_infos[idWord]["prop_end"], 59, color="grey", alpha=0.5)
    plt.xlim([0,59])
    plt.xticks(fontsize=13,ticks=[i for i in range(60)], labels=months, rotation=90)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig(output_data+idWord+"_fit.png")


def percentPhase(tweets, idWord, labels):
    plt.figure(figsize=[16, 4])
    plt.title("Répartition des clusters par phase (innovation(1) - propagation(2) - fixation|déclin(3))", fontsize=15)

    clustersByPhase = {}
    for ind, e in enumerate(tweets):
        month = e["day"][:7]

        if months.index(month) < dic_infos[idWord]["prop_beg"]:
            period = "phase 1"
        elif months.index(month) >= dic_infos[idWord]["prop_beg"] and months.index(month) < dic_infos[idWord]["prop_end"]:
            period = "phase 2"
        else:
            period = "phase 3"

        if period not in clustersByPhase:
            clustersByPhase[period] = []
        clustersByPhase[period].append(labels[ind])

    df = pd.DataFrame()
    for phase in clustersByPhase:
        for label in set(labels):
            try:
                df.loc[phase, label] = (len(np.where(np.array(clustersByPhase[phase]) == label)[0]) / len(
                    clustersByPhase[phase])) * 100
            except:
                df.loc[phase, label] = 0
    df = df.transpose()
    plt.bar(x=[i - 0.15 for i in range(len(df))], width=0.15, height=df.loc[:, "phase 1"].values.tolist(),
            label="Phase 1", color="#6C8CB2", zorder=2)
    plt.bar(x=[i for i in range(len(df))], width=0.15, height=df.loc[:, "phase 2"].values.tolist(), label="Phase 2",
            zorder=2, color="#B56E69")
    plt.bar(x=[i + 0.15 for i in range(len(df))], width=0.15, height=df.loc[:, "phase 3"].values.tolist(),
            label="Phase 3", zorder=2, color="#6B9E83")
    plt.legend(fontsize=14)
    plt.grid(axis="y", zorder=0)
    plt.xticks(ticks=[i for i in range(len(df))], labels=["Cluster " + str(c) for c in df.index.astype(str).tolist()],
               fontsize=14)
    plt.ylim([0, 100])
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(output_data+idWord+"_percentByPhase.png")


def topn_mostSimilar(meanVector, vectors, n):

    dic_similarity = {}
    for k in vectors :
        dic_similarity[k] = 1 - scipy.spatial.distance.cosine(meanVector, vectors[k])
    dic_similarity = sorted(dic_similarity.items(), key=lambda t: t[1], reverse=True)

    return [e[0] for e in dic_similarity[:n]]

def tweets_sample_firstAndLast(tweets,allEmb) :

    emb_firstMonth = {i:allEmb[i] for i, t in enumerate(tweets) if t["day"][:7] == "2014-01"}
    emb_lastMonth = {i:allEmb[i] for i, t in enumerate(tweets) if t["day"][:7] == "2018-12"}

    mean_emb_firstMonth = np.mean(list(emb_firstMonth.values()), axis=0)
    mean_emb_lastMonth = np.mean(list(emb_lastMonth.values()), axis=0)

    mostSimilarFirstMonth = topn_mostSimilar(mean_emb_firstMonth,emb_firstMonth,10)
    mostSimilarLastMonth = topn_mostSimilar(mean_emb_lastMonth,emb_lastMonth,10)

    first_tweets = np.array([t["tweet"] for i,t in enumerate(tweets) if i in mostSimilarFirstMonth])
    last_tweets = np.array([t["tweet"] for i,t in enumerate(tweets) if i in mostSimilarLastMonth])

    return [first_tweets, last_tweets]

def examples(tweets, idWord, labels) :
    out = open(output_data+idWord+"_examples.txt", "w")
    for cluster in set(labels) :
        clusterMembers = np.where(labels==cluster)[0]
        sample = np.random.choice(clusterMembers,25)
        out.write("\t\t\t<h2 id=\"cluster"+str(cluster)+"\">Cluster "+str(cluster)+" - "+str(len(clusterMembers))+" occurrences</h2>\n\t\t\t<ul>\n")
        for i in sample :
            tweet = tweets[i]["tweet"].replace("\n","<br/>")
            tweet = tweet.replace("<","&lt;")
            tweet = tweet.replace(">","&gt;")
            out.write("\t\t\t\t<li>"+tweet+"</li>\n")
        out.write("\t\t\t</ul>\n")
    out.close()

def globalView(idWord) :

    logger.info("Generating graphics for wordID " + idWord)

    allEmb = np.load(path_in+idWord+"_allEmb.npz", allow_pickle=True)["arr_0"]
    labels = np.load(path_in+idWord+"_labels.npy")
    tweets = [ujson.loads(line) for line in open(path_in+idWord+"_tweets.json")]

    viewClusters(allEmb, idWord, labels)
    fit(allEmb, idWord, labels)
    percentPhase(tweets, idWord, labels)

    logger.info("Generating graphics for wordID "+idWord+" - ended.")

    logger.info("Retrieving tweets from candidate word " + idWord)

    examples(tweets, idWord, labels)

    logger.info("Retrieving tweets from candidate word " + idWord + " - ended.")


toTreat = list(dic_infos.keys())
try :
    pool = Pool(processes=32)
    pool.map(globalView, toTreat)
finally:
    pool.close()
    pool.join()

logger.info("Generating pdf files.")

#
header = "<!DOCTYPE html>\n<html lang='fr'>\n\t<head>\n\t\t<meta charset='utf-8'/>\n\t\t<title></title>\n\t\t<link href=\"./style.css\" rel=\"stylesheet\"/>\n\t</head>\n\t<body>\n"

for i, idWord in enumerate(toTreat):

    logger.info("Generating pdf files for idWord "+idWord)

    allEmb = np.load(path_in+idWord+"_allEmb.npz", allow_pickle=True)["arr_0"]
    tweets = [ujson.loads(line) for line in open(path_in + idWord + "_tweets.json")]

    word = idByWord_inv[idWord]

    if dic_infos[idWord]["type"] == "change":

        first_tweets, last_tweets = tweets_sample_firstAndLast(tweets, allEmb)

        firstTweets = [tweet.replace("<", "&lt;") for tweet in first_tweets]
        firstTweets = [tweet.replace(">", "&gt;") for tweet in firstTweets]
        firstTweets = [tweet.replace("<br/>", "\n") for tweet in firstTweets]
        firstTweets = "".join(["\t\t\t\t<li>" + tweet + "</li>\n" for tweet in firstTweets])
        firstTweets = "\t\t\t\t<li class=\"first\">Tweets premier mois (2014-01)</li>\n" + firstTweets
        firstTweets = "\t\t\t<ul>\n" + firstTweets + "\t\t\t</ul>\n"

        lastTweets = [tweet.replace("<", "&lt;") for tweet in last_tweets]
        lastTweets = [tweet.replace(">", "&gt;") for tweet in lastTweets]
        lastTweets = [tweet.replace("<br/>", "\n") for tweet in lastTweets]
        lastTweets = "".join(["\t\t\t\t<li>" + tweet + "</li>\n" for tweet in lastTweets])
        lastTweets = "\t\t\t\t<li class=\"first\">Tweets dernier mois (2018-12)</li>\n" + lastTweets
        lastTweets = "\t\t\t<ul>\n" + lastTweets + "\t\t\t</ul>\n"

    clusters = []
    exampleFile = open(output_data+idWord+"_examples.txt")
    for line in exampleFile:
        if re.match(".*<h2 .* - .*</h2>.*", line):
            c = re.match(".*<h2 .*>(.*) - .*</h2>.*", line).group(1)
            clusters.append(c)
    menu = "\t\t<div class=\"menu\">\n\t\t\t<ul>\n"
    menu += "\t\t\t\t<li id=\"word\">" + word + "</li>\n\t\t\t\t<li> <a href=\"#graph\">Projection et courbe</a></li>\n"
    if dic_infos[idWord]["type"] == "change":
        menu += "\t\t\t\t<li> <a href=\"#globalView\">Échantillon début/fin de période</a></li>\n"
    menu += "\t\t\t\t<li> <a href=\"#clusterGraph\">Graph des clusters</a></li>\n"
    for c in clusters:
        nb = re.match("Cluster (.*)", c).group(1)
        menu += "\t\t\t\t<li><a href=\"#cluster" + nb + "\">" + "Cluster " + nb + "</a></li>\n"
    menu += "\t\t\t</ul>\n\t\t</div>\n"

    pathCluster = output_data + idWord + "_clusterScatter.png"
    pathFit = output_data + idWord + "_fit.png"
    pathPhase = output_data + idWord + "_percentByPhase.png"
    examples = open(output_data + idWord + "_examples.txt").read()
    examples = examples.replace("<br/>", "\n")
    titre = "\t\t<h1>" + word + " (" + dic_infos[idWord]["type"] + ")</h1>\n"
    part1 = "\t\t<div id=\"graph\">\n\t\t\t<h2>Projection clusters et alignement de courbe</h2>\n\t\t\t<img id=\"img1\" src='" + pathCluster + "' alt='Clustering - vue globale'/>\n\t\t\t<img id=\"img2\" src='" + pathFit + "' alt='Ajustement de courbe'/>\n\t\t</div>\n"
    part2 = "\t\t<div id=\"globalView\">\n\t\t\t<h2>Aperçu global - tweets de début et fin de période</h2>\n" + firstTweets + lastTweets + "\t\t</div>\n"
    part3 = "\t\t<h2 id=\"clusterGraph\">Répartition des clusters</h2>\n\t\t<img id=\"img3\" src='" + pathPhase + "' alt='Répartition des clusters par phase'/>\n"
    part4 = "\t\t<div class=\"clusters\">\n" + examples + "\n\t\t</div>\n\t</body>\n</html>"
    if dic_infos[idWord]["type"] == "change":
        html = header + menu + titre + part1 + part2 + part3 + part4
    else:
        html = header + menu + titre + part1 + part3 + part4
    outFile = open(output_html + idWord + ".html", "w")
    outFile.write(html)
    outFile.close()

    file = output_html+idWord+".html"
    os.system("pandoc " + file + " -o " + output_pdf + os.path.basename(file).split('.')[
            0] + ".pdf --pdf-engine=lualatex -V mainfont='DejaVu Sans' -V geometry:margin=0.5in")

    logger.info("Generating pdf files for idWord " + idWord+" - ended.")

logger.info("Generating pdf files - ended.")

logger.info("Ended.")