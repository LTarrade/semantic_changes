from multiprocessing import Pool, Manager
import numpy as np
import argparse
import logging
import ujson
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("--path_datasets", type=str, help="path to the datasets", required=True)
parser.add_argument("--path_emb", type=str,
                    help="path to the directory containing the embeddings (npz files mirrored with datasets)",
                    required=True)
parser.add_argument("--path_idByWord", type=str, help="path to the dictionary containing the ID of words",
                    required=True)
parser.add_argument("--path_candidates", type=str, help="path to the dictionary containing candidates", required=True)
parser.add_argument("--path_out", type=str, help="path to the output directory", required=True)

args = parser.parse_args()

path_datasets = args.path_datasets
path_emb = args.path_emb
path_idByWord = args.path_idByWord
path_candidates = args.path_candidates
path_out = args.path_out

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('embByWord_retrieval.log')
handler.setFormatter(logging.Formatter("%(asctime)s; %(levelname)s; %(message)s"))
logger.addHandler(handler)

logger.info("path_datasets: {}".format(path_datasets))
logger.info("path_emb: {}".format(path_emb))
logger.info("path_candidates: {}".format(path_candidates))
logger.info("path_out: {}".format(path_out))


def recupEmbByDay(day, lock, temp_files, tweet_files):
    logger.info("Processing day " + day)

    files = glob.glob(path_datasets + day + "*")
    fileNames = [os.path.basename(file).split(".")[0] for file in sorted(files)]

    for file in fileNames:

        f = open(path_datasets + file + ".json")
        emb = np.load(path_emb + file + "_emb.npz", allow_pickle=True)["arr_0"]

        tweets_dic = {}
        emb_dic = {}

        for l, line in enumerate(f):

            tweet = ujson.loads(line)
            tweet_wordEmb = tweet["words_emb"]
            user_id = tweet["user_id"]
            tweet_txt = tweet["tweet"]

            for w, word in enumerate(tweet_wordEmb):

                if word in wordsToTreat:

                    idWord = str(idByWord[word])
                    if idWord not in emb_dic:
                        emb_dic[idWord] = []
                        tweets_dic[idWord] = []

                    try:
                        emb_dic[idWord].append(emb[l][w].tolist())
                        tweets_dic[idWord].append({"tweet": tweet_txt, "user_id": user_id, "day": day})
                    except:
                        logger.info("Pb for " + idWord + " in " + file + ", line " + str(l))

        for idWord in emb_dic:
            with lock:
                out = open(temp_files[idWord], "a")
                for emb in emb_dic[idWord]:
                    out.write(ujson.dumps(emb) + "\n")
                out.close()

                out = open(tweet_files[idWord], "a")
                for t in tweets_dic[idWord]:
                    out.write(ujson.dumps(t) + "\n")
                out.close()

    logger.info("Processing day " + day + " - ended.")


logger.info("Loading data")

idByWord = ujson.load(open(path_idByWord))

dic = ujson.load(open(path_candidates))
wordsToTreat = set([dic[k]["word"] for k in dic])

months = [f"{y}-{m:02d}" for y in range(2014, 2019) for m in range(1, 13)]
days = sorted(list(set([os.path.basename(fileName).split(".")[0][:10] for fileName in glob.glob(path_datasets + "*")])))
days = [d for d in days if d[:7] in months]

temp_files = {idWord: path_out + idWord + "_emb_TEMP.json" for idWord in dic.keys()}
tweet_files = {idWord: path_out + idWord + "_tweets.json" for idWord in dic.keys()}

logger.info("Loading data - ended.")

manager = Manager()
lock = manager.Lock()

try:
    pool = Pool(processes=63)
    pool.starmap(recupEmbByDay, [(day, lock, temp_files, tweet_files) for day in days])
finally:
    pool.close()
    pool.join()

logger.info("Save emb in npz files")
for idWord in temp_files:
    try:
        file = open(temp_files[idWord])
        embeddings = []
        for line in file:
            emb = ujson.loads(line)
            embeddings.append(emb)
        np.savez_compressed(path_out + idWord + "_allEmb", embeddings)
        os.remove(path_out + temp_files[idWord])
    except:
        logging.info("no file for " + idWord)
logger.info("Save emb in npz files - ended")

logger.info("Ended.")
