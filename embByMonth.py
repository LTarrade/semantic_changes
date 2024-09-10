from multiprocessing import Pool
import numpy as np
import logging
import ujson
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("--pathDatasets", type=str, help="path to the datasets", required=True)
parser.add_argument("--pathEmb", type=str,
                    help="path to the directory containing the embeddings (npz files mirrored with datasets)",
                    required=True)
parser.add_argument("--pathID", type=str, help="path to the dictionary containing the ID of words", required=True)
parser.add_argument("--pathOut", type=str, help="path to the output directory", required=True)

args = parser.parse_args()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('embByMonth.log')
handler.setFormatter(logging.Formatter("%(asctime)s; %(levelname)s; %(message)s"))
logger.addHandler(handler)

pathDatasets = args.pathDatasets
pathID = args.pathID
pathEmb = args.pathEmb
pathOut = args.pathOut

logger.info("Loading data")
idByWord = ujson.load(open(pathID))
logger.info("Loading data - ended.")

months = sorted(list(set([os.path.basename(fileName).split(".")[0][:7] for fileName in
                          glob.glob(pathDatasets + "*")])))


def recupEmbByMonth(month):
    logger.info("Processing month " + month)

    files = glob.glob(pathDatasets + month + "*")
    fileNames = [os.path.basename(file).split(".")[0] for file in sorted(files)]
    dic = {}

    for file in fileNames:
        tweets_wordEmb = [ujson.loads(line)["words_emb"] for line in
                          open(pathDatasets + file + ".json")]
        emb = np.load(pathEmb + file + "_emb.npz", allow_pickle=True)["arr_0"]

        for l, wordList in enumerate(tweets_wordEmb):
            for w, word in enumerate(wordList):
                idWord = idByWord[word]
                if idWord not in dic:
                    dic[idWord] = []
                try:
                    dic[idWord].append(emb[l][w])
                except:
                    logger.info("missing words file " + file + " line " + str(l))

    for idWord in dic:
        dic[idWord] = np.mean(dic[idWord], axis=0)

    wordsLocation = list(dic.keys())
    emb = list(dic.values())

    np.savez_compressed(pathOut + month + "_wordsLocation",
                        wordsLocation)
    np.savez_compressed(pathOut + month + "_emb", emb)

    logger.info("Processing month " + month + " - ended.")


try:
    pool = Pool(processes=32)
    pool.map(recupEmbByMonth, months)
finally:
    pool.close()
    pool.join()

logger.info("Ended.")

"""
 ---------------------------
| ->->->-> by word <-<-<-<- |
 ---------------------------
 
embFiles = sorted(glob.glob(pathOut+"*emb.npz"))
locationFiles = sorted(glob.glob(pathOut+"*Location.npz"))

months = sorted(list(set([os.path.basename(file).split(".")[0].split('_')[0] for file in embFiles])))

words=set()
for i,file in enumerate(locationFiles) : 
    temp = np.load(file, allow_pickle=True)["arr_0"]
    for e in temp :
        words.add(e)
idWords = list(words)

dic = {k:{m:[] for m in months} for k in idWords}

for i,month in enumerate(months) :
    location = np.load([file for file in locationFiles if month in file][0], allow_pickle=True)["arr_0"]
    emb = np.load([file for file in embFiles if month in file][0], allow_pickle=True)["arr_0"]
    for w,word in enumerate(location) : 
        dic[word][month] = emb[w]

pathOut_2 = pathOut+"meanEmbByWord_month/"
out = open(pathOut_2+"months_ordered.nfo", "w")
for m in months : 
    out.write(m+"\n")
out.close()

for i,w in enumerate(dic) :
    sys.stdout.write("\r"+str(i+1))
    embeddings = []
    for m in dic[w] : 
        embeddings.append(dic[w][m])
    np.savez_compressed(pathOut_2+str(w)+"_byMonth_emb", embeddings)  
"""
