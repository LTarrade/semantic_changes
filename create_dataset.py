from multiprocessing import Pool,cpu_count
import argparse
import tarfile
import logging
import ujson
import glob
import sys
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="path to the compressed files", required=True)
parser.add_argument("--output", type=str, help="path to the directory where the dataset will be created", required=True)
parser.add_argument("--out_tok", type=str, help="path to the directory where the tokenized tweets are", required=True)
parser.add_argument("--occForms", type=str, help="path to the dictionary containing the number of occurrences of the words", required=True)
parser.add_argument("--pathID", type=str, help="path to the dictionary containing the ID of words", required=True)

args = parser.parse_args()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('create_dataset_allCorpusAn.log')
handler.setFormatter(logging.Formatter("%(asctime)s; %(levelname)s; %(message)s"))
logger.addHandler(handler)

path = args.path
output = args.output
out_tok = args.out_tok
occByWord = args.occForms
path_idByWord = args.pathID

idByWord = ujson.load(open(path_idByWord))
idByWord_inv = {str(v):k for k,v in idByWord.items()}

monthsToTreat = [f"{y}-{m:02d}" for y in range(2012, 2020) for m in range(1, 13)][2:-10]
wordsToKeep = set([k for k,v in occByWord.items() if (v>10000 and v<200000) and (re.match("^[\w'-]+$", idByWord_inv[k])) and (not re.match("^\d+$", idByWord_inv[k]))])

logger.info(str(len(wordsToKeep))+" words to retrieve emb")

logger.info("path : %s, output : %s, months : %s -> %s"%(path,output,monthsToTreat[0], monthsToTreat[-1]))

def process(month) : 
    logger.info("month "+month)
    files = getFilesToTreat(path, month)
    concatenateAndFormat(files)
    logger.info(" %s - ended."%month)

# Function that retrieves compressed files containing the tweets for the given month
def getFilesToTreat(path, month):

    files=[]

    for i,fileName in enumerate([f for f in glob.glob(path + '*.tgz') if f.split('/')[-1][:7] in month]):
        tf=tarfile.open(fileName)
        files+=[(n,tf) for n in tf.getnames()]
        
    files.sort()

    return files

# function that retrieves the tokenization of tweets
def retrieveTokenizedTweets(file) : 
    
    tokenized_tweets = {}
    to_remplace = {}

    for line in file : 
        
        tokenized_tweet = ujson.loads(line)

        to_remplace[tokenized_tweet["id"]]=[]

        # we retrieve the tokenization already done
        words = []
        for k in tokenized_tweet["tokenization"] : 
            words+=[w.lower() for w in tokenized_tweet["tokenization"][k]["tokens"]]
        for i,w in enumerate(words) :
            if w.startswith("http:") or w.startswith("https:") or w.startswith("www.") : 
                words[i]="<URL>"
                to_remplace[tokenized_tweet["id"]].append((w,"<URL>"))
            if re.match(r"@\w+", w) : 
                words[i]="<MENTION>"
                to_remplace[tokenized_tweet["id"]].append((w,"<MENTION>"))
        tokenized_tweets[tokenized_tweet["id"]] = words

    return (tokenized_tweets,to_remplace)


# Function that creates the dataset according to the given files
def concatenateAndFormat(files) : 
    
    day_prec = ""
    
    for i,file in enumerate(files) : 
        
        f = file[1].extractfile(file[0])
        name = os.path.basename(file[0])
        day = name[:10]

        if day!=day_prec :
            if i!=0 :
                logger.info("Concatenation complete for %s: %d tweets grouped into %d file(s)"%(day_prec, nbTweets, nbFile))
            out = open(output+day+".json", "a")
            nbFile=1
            nbFile_ext=""
            nbTweets=0
            
        logger.info("File %s (%d/%d)"%(name, i+1, len(files)))
        
        out = open(output+day+nbFile_ext+".json", "a")

        correspondingFile_tokenized = open(out_tok+day+nbFile_ext+"_tokenized.json")
        tokenized_tweets, to_remplace = retrieveTokenizedTweets(correspondingFile_tokenized)

        for j,line in enumerate(f) : 

            tweet_object = ujson.loads(line)
            tweet = tweet_object["tweet"]
            tweet_id = tweet_object["id"]
            user_id = tweet_object["user"]["id"]
            date = tweet_object["date"][:10]

            words = tokenized_tweets[tweet_id]

            words_emb = [w for w in words if w in wordsToKeep] 
            
            for pair in to_remplace[tweet_id] : 
                tweet = re.sub(re.escape(pair[0]), pair[1], tweet, flags=re.I)
            
            dic = {"tweet":tweet, "tweet_id":tweet_id, "user_id":user_id, "date":date, "words":words, "words_emb":words_emb}
            out.write(ujson.dumps(dic)+"\n")
            nbTweets+=1
            
            if nbTweets%100000==0 :
                nbFile+=1
                nbFile_ext="_"+str(nbFile)
                out.close()
                out = open(output+day+nbFile_ext+".json", "a")
                correspondingFile_tokenized = open(out_tok+day+nbFile_ext+"_tokenized.json")
                tokenized_tweets, to_remplace = retrieveTokenizedTweets(correspondingFile_tokenized)
                
        if (i+1==len(files)) :
            logger.info("Concatenation complete for %s: %d tweets grouped into %d file(s)"%(day, nbTweets, nbFile))
            
        out.close() 
        day_prec = day

try :
    pool = Pool(processes=32)
    pool.map(process, monthsToTreat)
finally:
    pool.close()
    pool.join()

logger.info("Ended.")
