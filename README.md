This directory contains the scripts used to retrieve contextual embeddings of target words, select candidates for semantic change, and generate the data needed to evaluate the results. 

1. **create_dataset.py:** This script is a pre-processing script that formats the corpus as a dataset to speed up processing when calculating embeddings. 
<br><br>It takes as input the path to the compressed corpus and its tokenised version, and the dictionaries containing respectively the number of occurrences of each word in the corpus and the identifiers associated with each of them. 
<br><br>In the output, there are as many datasets as there are files. Each file is limited to 100,000 tweets, and is named according to the day on which the tweets were posted. They are composed of one tweet per line in json format with the following keys: 
	- tweet (the tweet)
	- tweet_id (its identifier)
	- user_id (the identifier of the author of the tweet)
	- words (the tweet in its tokenised form)
	- words_emb (the tokens for which embeddings need to be retrieved).


2. **embedding_retrieval.py:** This script retrieves the vectors of the target words. 
<br><br>It takes as input the path to the datasets containing the tokenisation of the tweets and the words whose vectors are to be retrieved.
<br><br>As output, it provides files mirroring those in the dataset, in *.npz* format, containing the vectors. For example, the fourth word (index 3) in the list of target words present in the tweet on line 359 (index 358) of the 2015-03-04_2.json file can be retrieved at index \[358\]\[3\] in the array contained in the 2015-03-04_2_emb.npz file.


3. **embByMonth.py:** This script retrieves the average vectors per month.
<br><br>It takes as input the datasets and the npz files mirroring the datasets containing the embeddings. 
<br><br>The output is the average embeddings for each target word each month (+ modified file organisation). The end result is one file per word (file name = word identifier), with a *.nfo* file indicating the index for each month in the files.


4. **select_semanticChange_candidates.py:** This script selects likely semantic changes. 
<br><br>It takes as input the average embeddings per month of each word 
<br><br>He provides as output : 
	- *data/df_5years_dist.csv:* for each form, the cosine distances over 5 years of each month from the first month.
    - *data/df_5years_dist_rel.csv:* same as above but with relative distances.
	- *data/lmfit_log_results_dist.csv:* the results of fitting these trajectories with a logistic function.
	- *data/lmfit_logNorm_results_dist.csv:* the results of fitting these trajectories with a lognormal function.
	- *data/candidates.json:* dictionary containing all the selected candidates, identified by their ID, with its type, the values of the fitted trajectory, the start and end of its propagation period, and the corresponding word.


5. **embByWord_retrieval.py :** This script retrieves all the embeddings of the candidates for semantic change. 
<br><br>Input :
	- the path of the directory where the datasets are stored
    - the directory where the embeddings are stored (mirrored *.npz*)
    - the dictionary containing the word identifiers
    - the dictionary containing information about the selected semantic changes
    - the output directory
<br><br>Output : 
	- one *json* file per word, one line per embedding, containing metadata about each embedding (tweet, user id, tweet date (ddd-mm-dd))
    - one *npz* file per word, containing all its vectors, mirrored with the json file (index of the vector = index of the corresponding tweet)


6. **clustering.py :** This script performs clustering on all the embeddings of a word to recover its main usages (for all words).
<br><br>Input: it takes the path to the *.npz* files containing the embeddings for each word.
<br><br>Output: a numpy file for each word, containing the labels associated with each embedding (mirrored indices with the *.npz* of the embeddings and the *.json* of the tweets).


8. **visualize_candidates.py :** This script generates pdf files to help with the decision to evaluate candidates for semantic change.
<br><br>Input: It takes as input the dictionary containing information on the semantic changes selected, as well as the related files containing the tweets in which they appear, their embeddings and the cluster to which each belongs. It also takes as input the dataframe containing the (relative) cosine distances of each month's average embedding from that of the first month, and the dictionary containing the word identifiers. 
<br><br>Output: The pdf files contain different visualisations as well as a random sample of tweets corresponding to each cluster ([here](https://perso.ens-lyon.fr/louise.tarrade/candidats_semanticShift/html/))


