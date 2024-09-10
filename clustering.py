from multiprocessing import Pool
import sklearn.cluster
import pandas as pd
import numpy as np
import logging
import argparse
import sklearn
import umap
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("--path_emb", type=str,
                    help="path to the directory containing the embeddings (npz files mirrored with datasets)",
                    required=True)
parser.add_argument("--path_out", type=str, help="path to the output directory", required=True)
args = parser.parse_args()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('clustering.log')
handler.setFormatter(logging.Formatter("%(asctime)s; %(levelname)s; %(message)s"))
logger.addHandler(handler)


def retrieveEmb(file):
    embeddings = np.load(file, allow_pickle=True)["arr_0"]
    df = pd.DataFrame.from_records(embeddings)
    return df


def clustering(df, dim, n, d, c, s):
    reducer = umap.UMAP(n_components=dim, n_neighbors=n, min_dist=d, random_state=42)
    umap_result = reducer.fit_transform(df.values)
    df_temp = pd.DataFrame(umap_result, index=df.index)
    X = df_temp.values

    hdb = sklearn.cluster.HDBSCAN(min_cluster_size=c, min_samples=s)
    clustering = hdb.fit(X)
    labels = clustering.labels_

    return labels


def retrieveCluster(file):
    idWord = os.path.basename(file).split("_")[0]
    logger.info("Clustering : " + idWord)

    df_allEmb = retrieveEmb(file)
    labels = clustering(df_allEmb, dim=10, n=int(0.005 * len(df_allEmb)), d=0.05, c=int(0.01 * len(df_allEmb)),
                        s=int(0.0075 * len(df_allEmb)))

    np.save(path_out + idWord + "_labels", labels)
    logger.info("Clustering : " + idWord + " - ended.")

path_emb = args.path_emb
path_out = args.path_out

files = glob.glob(path_emb + "*_allEmb.npz")

try:
    pool = Pool(processes=30)
    pool.map(retrieveCluster, files)
finally:
    pool.close()
    pool.join()

logger.info("ended.")
