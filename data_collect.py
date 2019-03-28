import json
import os
import numpy as np
import multiprocessing
from collections import defaultdict
from copy import copy
import nltk
nltk.download('punkt')
from joblib import Parallel, delayed
import multiprocessing
import string
from threading import Thread
import tqdm
import implicit #use conda install -c conda-forge implicit
from scipy.sparse import csr_matrix, find, lil_matrix, dok_matrix
import time
from colorama import Fore, Back, Style
from sklearn import metrics as skmet
import scipy

import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class PopularityRanking:
    def __init__(self):
        self.scores = None

    def fit(self, train):
        self.scores = np.sum(train, axis=1)
    def recommend(self, row=0, interaction_matrix=0, N=1, recalculate_user=False): #parameters just to fit interface
        to_return = []
        max = None
        for i in range(N):
            max_index = np.argmax(self.scores)
            score = self.scores[max_index,0]
            if score == -1:
                break
            if max is None:
                max = score
            self.scores[max_index] = -1
            to_return.append((max_index, float(score)/max))
        return to_return

class Playlist:
    def __init__(self, name, collaborative, pid, modified_at, num_tracks, num_albums, num_followers, tracks):
        self.name = name
        self.collaborative = collaborative
        self.pid = pid
        self.modified_at = modified_at
        self.num_tracks = num_tracks
        self.num_albums = num_albums
        self.num_followers = num_followers
        self.tracks = tracks
        self.features = {}


class Track:
    def __init__(self, artist_name, track_uri, artist_uri, track_name, album_uri, duration_ms, album_name):
        self.artist_name = artist_name
        self.track_uri = track_uri
        self.artist_uri = artist_uri
        self.track_name = track_name
        self.album_uri = album_uri
        self.duration_ms = duration_ms
        self.album_name = album_name
        self.features = {}


class PlaylistTrack:
    def __init__(self, track, pos):
        self.track = track
        self.pos = pos


class SparseMatrix:
    def __init__(self, num_rows, num_cols, entries=None):
        # NOTE: entries are (row, col, value)
        self.num_rows = num_rows
        self.num_cols = num_cols

        self.rows = {}

        if entries is not None:
            for row, col, value in entries:
                if row not in self.rows:
                    self.rows[row] = defaultdict(lambda: 0)
                self.rows[row][col] = value

    def setValue(self, row, col, value):
        if row not in self.rows:
            self.rows[row] = defaultdict(lambda: 0)
        self.rows[row][col] = value

    def getValue(self, row, col):
        if row in self.rows:
            return self.rows[row][col]
        return 0

    def getRow(self, row):
        if row in self.rows:
            to_return = np.zeros((self.num_cols,))
            for col, value in self.rows[row].items():
                to_return[col] = value
            return to_return
        return None

    def getCol(self, col):
        colVec = np.zeros(self.num_rows)
        for ind, ints in self.rows.items():
            colVec[ind] = ints[col]
        return colVec

    def printMatrix(self):
        for row in range(self.num_rows):
            print(self.getRow(row))

    def getInteractions(self):
        interactions = []
        for row, vals in self.rows.items():
            for col, val in vals.items():
                interactions.append((row, col, val))
        return interactions

    def getTransposeInteractions(self):
        interactions = []
        for row, vals in self.rows.items():
            for col, val in vals.items():
                interactions.append((col, row, val))
        return interactions

    def getRowInteractions(self, row):
        interactions = []
        for interaction in self.rows[row].items():
            interactions.append(interaction)
        return interactions

    def getTranspose(self):
        newMatr = SparseMatrix(self.num_cols, self.num_rows, self.getTransposeInteractions())
        return newMatr


class Artist:
    def __init__(self, name, uri, popularity):
        self.name = name
        self.uri = uri
        self.popularity = popularity

    def __str__(self):
        return "Name: " + self.name + ", URI: " + self.uri + ", Popularity: " + str(self.popularity)


METRIC_DICT = {}
METRICS = ["NDCG", "RPrec", "Prec@1"]
def output_latex():
    global METRIC_DICT, local_artists, ALGORITHMS
    num_col = (len(METRIC_DICT)+3)
    num_algorithms = len(ALGORITHMS)
    output_str = ""
    for level in ["artist", "track"]:
        output_str += "\\begin{table*}\n\\begin{tabular}{" + ("| c " * num_col) + "|}"
        if level == "artist":
            name = "Artists"
        else:
            name = "Tracks"
        output_str += "\n\\hline\n\\multicolumn{" + str(num_col) + "}{|l|}{"+name+"} \\\\"
        output_str += "\n\\hline\n& & "
        for key in METRIC_DICT.keys():
            output_str += key + " & "
        output_str += "Average \\\\"
        output_str += "\n\\hline\n\\multicolumn{2}{|l|}{Local "+name+"}"
        vals = []
        for key in METRIC_DICT.keys():
            if level == "artist":
                the_col = local_artists_found[key]
            else:
                the_col = local_tracks_by_city[key]
            length = len(the_col)
            output_str += " & " + str(length)
            vals.append(length)
        avg = np.mean(vals)
        output_str += " & " + "{:10.3f}".format(avg)
        output_str += " \\\\\n\\hline\n"
        for m in METRICS:
            output_str += "\\multirow{" + str(num_algorithms) + "}{*}{\\rotatebox[origin=c]{90}{"+m+"}} "
            for algo in ALGORITHMS:
                output_str += "& " + algo
                vals = []
                for city in METRIC_DICT.keys():
                    output_str += " & " + str(METRIC_DICT[city][algo][level][m])
                    vals.append(METRIC_DICT[city][algo][level][m+"VAL"])
                avg = np.mean(vals)
                output_str += " & " + str("{:10.3f}".format(avg))
                output_str += " \\\\\n\cline{2-" + str(num_col) + "}\n"
            output_str += "\\hline\n"
        output_str += "\\end{tabular}\n\\caption{Evaluation metrics at the " + level + " level for each algorithm.}\n\\end{table*}\n\n"
    print(output_str)



def r_precision(recs, correct, mode="artist", random=0):
    global tracks, track_ids, local_artists, city_to_test
    if mode=="artist":
        remaining_artists = list(local_artists[city_to_test].keys()).copy()
        num_to_check = len(correct)
        num_total = len(recs)
        num_correct = 0
        i = 0
        i2 = 0
        while i < num_to_check and i2 < num_total:
            if random==0:
                rec, score = recs[i2]
            else:
                rec = recs[i2]
            track = tracks[track_ids[rec]]
            if track.artist_uri in remaining_artists:
                if track.artist_uri in correct:
                    num_correct += 1
                i += 1
                remaining_artists.remove(track.artist_uri)
            i2 += 1
        return float(num_correct) / float(num_to_check)
    elif mode=="track":
        num_to_check = len(correct)
        num_total = len(recs)
        num_correct = 0
        i = 0
        i2 = 0
        while i < num_to_check and i2 < num_total:
            rec, score = recs[i2]
            track = tracks[track_ids[rec]]
            if track.artist_uri in local_artists[city_to_test]:
                if track.track_uri in correct:
                    num_correct += 1
                i += 1
            i2 += 1
        return float(num_correct) / float(num_to_check)

def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_full(r, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    k = len(r)
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def clicks(recs, correct, mode="artist"):
    global tracks, track_ids, local_artists, city_to_test
    if mode == "artist":
        remaining_artists = list(local_artists[city_to_test].keys()).copy()
        num_total = len(recs)
        i = 0
        i2 = 0
        while i2 < num_total:
            rec, score = recs[i2]
            track = tracks[track_ids[rec]]
            if track.artist_uri in remaining_artists:
                if track.artist_uri in correct:
                    return int(i / 10)
                i += 1
                remaining_artists.remove(track.artist_uri)
            i2 += 1
    elif mode == "track":
        remaining_artists = local_artists[city_to_test]
        num_total = len(recs)
        i = 0
        i2 = 0
        while i2 < num_total:
            rec, score = recs[i2]
            track = tracks[track_ids[rec]]
            if track.artist_uri in remaining_artists:
                if track.track_uri in correct:
                    return int(i / 10)
                i += 1
            i2 += 1


ALGORITHMS = ["ALS", "BPR", "Random", "Popular"]
local_data_path = "/Users/akimchukdaniel/Google Drive/locals.json"
local_artists = {}
local_artists_found = {}
local_tracks_by_city = {}
num_local_playlists = {}
local_data_file = open(local_data_path)
file_contents= local_data_file.read()
local_json = json.loads(file_contents)
for city in local_json:
    local_artists[city] = {}
    local_artists_found[city] = set()
    for artist in local_json[city]:
        local_artists[city][artist["artist_uri"]] = Artist(artist["artist_name"], artist["artist_uri"], artist["artist_popularity"])
eprint("Done importing local artists.")

for city in local_json:
    eprint(Back.GREEN, "RUNNING FOR CITY", city, Back.RESET)
    METRIC_DICT[city] = {}
    data_path = "/Users/akimchukdaniel/mpd_data/mpd.v1/data_big/"
    test_data_path = "/Users/akimchukdaniel/mpd_data/challenge.v1/challenge_set.json"
    city_to_test = city
    tracks = {}
    local_tracks = {}
    track_ids = []
    track_id_len = 0
    track_id_to_index = {}
    playlists = {}
    pids = []
    test_pids = []
    local_pids = []
    potential_eval_pids = []
    interactions = {}
    file_count = 0
    filenames = list(os.listdir(data_path))
    for i in range(len(filenames)):
        filenames[i] = data_path + filenames[i]
    filenames.append(test_data_path)

    for filename in tqdm.tqdm(filenames):
        if filename == ".DS_Store":
            continue
        if filename == test_data_path:
            num_playlists_train = num_playlists
            isTest = True
        else:
            isTest = False
        data_file = open(filename)
        file_count += 1
        file_contents = data_file.read()
        jsonArray = json.loads(file_contents)
        for playlist_data in jsonArray["playlists"]:
            is_local = False
            num_local = 0
            try:
                pid = int(playlist_data["pid"])

                try:
                    name = playlist_data["name"]
                except:
                    name = None

                try:
                    collab = playlist_data["collaborative"] == 'true'
                except:
                    collab = None

                try:
                    modified_at = int(playlist_data["modified_at"])
                except:
                    modified_at = None

                try:
                    num_tracks = int(playlist_data["num_tracks"])
                except:
                    num_tracks = None

                try:
                    num_albums = int(playlist_data["num_albums"])
                except:
                    num_albums = None

                try:
                    num_followers = int(playlist_data["num_followers"])
                except:
                    num_followers = None

                try:
                    tracks_data = playlist_data["tracks"]
                    playlist_interactions = defaultdict(lambda: 0)
                    playlist_tracks = []
                    for track_data in tracks_data:
                        track_uri = track_data["track_uri"]
                        if track_uri in tracks:
                            track = tracks[track_uri]
                        else:
                            artist_name = track_data["artist_name"]
                            artist_uri = track_data["artist_uri"]
                            track_name = track_data["track_name"]
                            album_uri = track_data["album_uri"]
                            duration_ms = track_data["duration_ms"]
                            album_name = track_data["album_name"]
                            track = Track(artist_name, track_uri, artist_uri, track_name, album_uri, duration_ms,
                                          album_name)
                            tracks[track_uri] = track
                            track_ids.append(track_uri)
                            track_id_to_index[track_uri] = track_id_len
                            track_id_len += 1
                            if artist_uri in local_artists[city_to_test]:
                                num_local += 1
                                is_local = True
                                # print(artist_name, "is local")
                                local_tracks[track_uri] = track
                                local_artists_found[city_to_test].add(artist_uri)
                                #print(track_name, "by", artist_name, "is local.")

                        try:
                            pos = int(track_data["pos"])
                        except:
                            pos = None
                        playlist_tracks.append(PlaylistTrack(track, pos))
                        playlist_interactions[track_uri] = playlist_interactions[track_uri] + 1
                except:
                    playlist_tracks = []
                    playlist_interactions = None

                playlist = Playlist(name, collab, pid, modified_at, num_tracks, num_albums, num_followers,
                                    playlist_tracks)
                playlists[pid] = playlist
                interactions[pid] = playlist_interactions
                pids.append(pid)
                if isTest:
                    test_pids.append(pid)
                if is_local:
                    # print(pid,"is a local playlist")
                    local_pids.append(pid)
                    potential_eval_pids.append(pid)
            except Exception as e:
                eprint(str(e))
                pass
        num_playlists = len(playlists)
        num_tracks = len(tracks)

    print("Imported " + str(num_playlists) + " playlists containing " + str(num_tracks) + " unique tracks from " + str(
        file_count) + " files.")
    print("Local Playlists: " + str(len(local_pids)))
    num_local_playlists[city_to_test] = len(local_pids)
    print("Local Artists:",len(local_artists[city_to_test]))
    print("Local Artists Found:", len(local_artists_found[city_to_test]))
    print("Local Tracks:",len(local_tracks))
    print()
    print()
    eprint("Imported " + str(num_playlists) + " playlists containing " + str(num_tracks) + " unique tracks from " + str(
        file_count) + " files.")
    eprint("Local Playlists: " + str(len(local_pids)))
    eprint("Local Artists:", len(local_artists[city_to_test]))
    eprint("Local Artists Found:", len(local_artists_found[city_to_test]))
    eprint("Local Tracks:", len(local_tracks))
    eprint()
    eprint()
    del filenames
    local_tracks_by_city[city_to_test] = local_tracks

    # interaction_matrix = SparseMatrix(num_playlists, num_tracks)
    interaction_matrix_rows = []
    interaction_matrix_cols = []
    interaction_matrix_vals = []
    # interaction_matrix = dok_matrix((num_playlists,num_tracks))

    row_count = 0
    test_indexes = []
    r_train = dok_matrix((num_playlists, num_tracks))
    num_to_pick = int(len(potential_eval_pids) / 10)
    eval_pids = np.random.choice(potential_eval_pids, num_to_pick)
    # if city_to_test == "Boulder":
    #     eval_pids = [778762, 794490, 410260, 926844, 778146, 739622, 375363, 943589, 370638, 244967,
    #      778146]
    # elif city_to_test == "Nashville":
    #     eval_pids = [683526, 438085, 201602, 325159 ,903413, 383142 ,445589, 399406, 324721]
    # else:
    #     eval_Pids = [  3850, 712986, 531600, 344761, 116566, 850254, 179278 ,596946 ,366215 ,673698,
    #  54515, 193907, 293899, 892507, 823139, 805938, 510990, 881103, 416453, 902981,
    # 497965, 841557, 371559, 156811,  75504, 391534, 509505, 384195, 319749, 642468,
    # 998061, 416453, 794321, 324031, 694307, 378391, 528356]
    print("Evaluation Playlist IDs:",eval_pids)
    eprint("Evaluation Playlist IDs:",eval_pids)

    correct_ids = {}
    correct_track_ids = {}
    for row in tqdm.tqdm(range(len(pids))):
        row_count += 1
        is_eval = pids[row] in eval_pids
        if is_eval:
            correct_ids[pids[row]] = []
            correct_track_ids[pids[row]] = []
        ints = interactions[pids[row]]
        if pids[row] in test_pids:
            test_indexes.append(row)
        for (track_id, count) in ints.items():
            index = track_id_to_index[track_id]
            # interaction_matrix.setValue(row, index, count)
            if not is_eval or tracks[track_id].artist_uri not in local_artists[city_to_test]:
                interaction_matrix_rows.append(row)
                interaction_matrix_cols.append(index)
                interaction_matrix_vals.append(count)
            if not is_eval:
                r_train[row, index] = count
            elif tracks[track_id].artist_uri in local_artists[city_to_test]:
                correct_ids[pids[row]].append(tracks[track_id].artist_uri)
                correct_track_ids[pids[row]].append(tracks[track_id].track_uri)
            # interaction_matrix[row,index] = count
        # for playlist_track in playlist.tracks:
        #    track_uri = playlist_track.track.track_uri
        #    col = track_id_to_index[track_uri]
        #    interaction_matrix[row,col] = 1

    del interactions
    interaction_matrix = csr_matrix((interaction_matrix_vals, (interaction_matrix_rows, interaction_matrix_cols)),
                                    shape=(num_playlists, num_tracks))
    eprint("Built interaction matrix for", row_count, "playlists.")
    eval_pids = list(dict.fromkeys(eval_pids).keys())


    for METHOD in ["Popular"]:
        METRIC_DICT[city_to_test][METHOD] = {"artist": {}, "track": {}}
        if METHOD == "Random":
            eprint(Fore.CYAN, "RANDOM SELECTION", Fore.RESET)
        else:
            if METHOD == "ALS":
                eprint(Fore.CYAN,"ALTERNATING LEAST SQUARES",Fore.RESET)
                model = implicit.als.AlternatingLeastSquares(factors=224, use_gpu=False)  ## power of 8 for gpu usage
            elif METHOD == "BPR":
                eprint(Fore.CYAN,"BAYESIAN PERSONALIZED RANKING",Fore.RESET)
                model = implicit.bpr.BayesianPersonalizedRanking(factors=224, use_gpu=False)
            elif METHOD == "Popular":
                eprint(Fore.CYAN,"POPULARITY BASELINE",Fore.RESET)
                model = PopularityRanking()

            model.fit(r_train.T)
        metric_list = []
        ndcg_list = []
        rec_list = []
        prec_at_one_list = []

        track_metric_list = []
        track_ndcg_list = []

        correct_list = []
        correct_track_list = []
        track_prec_at_one_list = []

        playlist_id = eval_pids[0]
        for playlist_id in eval_pids:
            if METHOD == "Random":
                random_recs = []
                for local in local_tracks.keys():
                    random_recs.append((track_id_to_index[local], np.random.rand()))
                    random_recs = sorted(random_recs, key=lambda x: x[1])

            playlist_metrics_x = []
            track_playlist_metrics_x = []
            playlist_metrics_actual = []
            track_playlist_metrics_actual = []
            remaining_artists = list(local_artists[city_to_test].keys()).copy()
            ndcg = []
            track_ndcg = []
            #print("pid:", playlist_id)
            row = pids.index(playlist_id)
            # print(row)
            # print(local_artists["Nashville"])
            if METHOD != "Random":
                recs = model.recommend(row, interaction_matrix, N=num_tracks, recalculate_user=METHOD=="ALS")
            else:
                recs = random_recs
            print(recs)
            rec_list.append(recs)
            correct_list.append(correct_ids[playlist_id])
            correct_track_list.append(correct_track_ids[playlist_id])
            interactions = interaction_matrix[row]
            #print("IN PLAYLIST")
            # for interaction in interactions.nonzero()[1]:
            #     track = tracks[track_ids[interaction]]

            #print("\nRECOMMENDS")
            count = 1

            track_count = 1
            #print(correct_track_ids[playlist_id])
            is_first = True
            for rec, score in recs:
                track = tracks[track_ids[rec]]
                if track.artist_uri in local_artists[city_to_test]:
                    track_playlist_metrics_x.append(score)
                    if track.track_uri in correct_track_ids[playlist_id]:
                        if is_first:
                            track_prec_at_one_list.append(1)
                        track_ndcg.append(1)
                        escape = Fore.GREEN
                        track_playlist_metrics_actual.append(1)
                    else:
                        if is_first:
                            track_prec_at_one_list.append(0)
                        track_ndcg.append(0)
                        track_playlist_metrics_actual.append(0)
                        if track.artist_uri in correct_ids[playlist_id]:
                            escape = Fore.YELLOW
                        else:
                            escape = Fore.RED
                    track_count += 1
                    #print(escape, track.track_uri,track.track_name, "by", track.artist_name, "score:", score, "AT POSITION:", count,Fore.RESET)




                if track.artist_uri in remaining_artists:
                    playlist_metrics_x.append(score)
                    if track.artist_uri in correct_ids[playlist_id]:
                        if is_first:
                            prec_at_one_list.append(1)
                            is_first=False
                        playlist_metrics_actual.append(1)
                        escape = Back.GREEN
                        ndcg.append(1)
                    else:
                        if is_first:
                            prec_at_one_list.append(0)
                            is_first=False
                        playlist_metrics_actual.append(0)
                        escape = Back.RED
                        ndcg.append(0)
                    #print(escape, track.track_name, "by", track.artist_name, "score:", score, "AT POSITION:", count)
                    remaining_artists.remove(track.artist_uri)
                    count += 1
            #print(Style.RESET_ALL + "\n\n\n")
            metrics = (playlist_metrics_x, playlist_metrics_actual)
            metric_list.append(metrics)
            ndcg_list.append(ndcg)

            track_metrics = (track_playlist_metrics_x, track_playlist_metrics_actual)
            track_metric_list.append(track_metrics)
            track_ndcg_list.append(track_ndcg)

        #ARTIST LEVEL
        auc_list = []
        r_prec_list = []
        ndcg_metric_list = []
        click_list = []

        for i in tqdm.tqdm(range(len(rec_list))):
            #auc = skmet.roc_auc_score(metric_list[i][1], metric_list[i][0])
            r_prec = r_precision(rec_list[i], correct_list[i])
            ndcg = ndcg_full(ndcg_list[i])
            #click_count = clicks(rec_list[i], correct_list[i])

            #auc_list.append(auc)
            r_prec_list.append(r_prec)
            ndcg_metric_list.append(ndcg)
            #click_list.append(click_count)

        eprint(Fore.RED, "ARTIST LEVEL", Fore.RESET)
        #eprint("AVERAGE AUC:", np.mean(auc_list),"STDERR:",scipy.stats.sem(auc_list), "STD:",np.std(auc_list))
        eprint("AVERAGE R_PRECISION:", np.mean(r_prec_list),"STDERR:",scipy.stats.sem(r_prec_list), "STD:",np.std(r_prec_list))
        eprint("AVERAGE NDCG:", np.mean(ndcg_metric_list),"STDERR:",scipy.stats.sem(ndcg_metric_list), "STD:",np.std(ndcg_metric_list))
        eprint("AVERAGE PREC@1:", np.mean(prec_at_one_list),"STDERR:",scipy.stats.sem(prec_at_one_list),"STD",np.std(prec_at_one_list))
        #eprint("AVERAGE CLICKS:", np.mean(click_list),"STDERR:",scipy.stats.sem(click_list), "STD:",np.std(click_list))
        METRIC_DICT[city_to_test][METHOD]["artist"]["RPrec"] = str("{:10.3f}".format(np.mean(r_prec_list))) + " (" + str("{:10.3f}".format(scipy.stats.sem(r_prec_list))) + ")"
        METRIC_DICT[city_to_test][METHOD]["artist"]["NDCG"] = str("{:10.3f}".format(np.mean(ndcg_metric_list))) + " (" + str("{:10.3f}".format(scipy.stats.sem(ndcg_metric_list))) + ")"
        METRIC_DICT[city_to_test][METHOD]["artist"]["Prec@1"] = str("{:10.3f}".format(np.mean(prec_at_one_list))) + " (" + str("{:10.3f}".format(scipy.stats.sem(prec_at_one_list))) + ")"

        METRIC_DICT[city_to_test][METHOD]["artist"]["RPrecVAL"] = np.mean(r_prec_list)
        METRIC_DICT[city_to_test][METHOD]["artist"]["NDCGVAL"] = np.mean(ndcg_metric_list)
        METRIC_DICT[city_to_test][METHOD]["artist"]["Prec@1VAL"] = np.mean(prec_at_one_list)

        #TRACK LEVEL
        auc_list = []
        r_prec_list = []
        ndcg_metric_list = []
        click_list = []

        for i in tqdm.tqdm(range(len(rec_list))):
            #auc = skmet.roc_auc_score(track_metric_list[i][1], track_metric_list[i][0])
            r_prec = r_precision(rec_list[i], correct_track_list[i], mode="track")
            ndcg = ndcg_full(track_ndcg_list[i])
            #click_count = clicks(rec_list[i], correct_track_list[i], mode="track")

            #auc_list.append(auc)
            r_prec_list.append(r_prec)
            ndcg_metric_list.append(ndcg)
            #click_list.append(click_count)
        eprint(Fore.RED, "TRACK LEVEL", Fore.RESET)
        # eprint("AVERAGE AUC:", np.mean(auc_list), "STDERR:", scipy.stats.sem(auc_list), "STD:", np.std(auc_list))
        eprint("AVERAGE R_PRECISION:", np.mean(r_prec_list), "STDERR:", scipy.stats.sem(r_prec_list), "STD:",
              np.std(r_prec_list))
        eprint("AVERAGE NDCG:", np.mean(ndcg_metric_list), "STDERR:", scipy.stats.sem(ndcg_metric_list), "STD:",
              np.std(ndcg_metric_list))
        eprint("AVERAGE PREC@1:", np.mean(track_prec_at_one_list),"STDERR:",scipy.stats.sem(track_prec_at_one_list),"STD",np.std(track_prec_at_one_list))

        # eprint("AVERAGE CLICKS:", np.mean(click_list), "STDERR:", scipy.stats.sem(click_list), "STD:",
        #       np.std(click_list))
        METRIC_DICT[city_to_test][METHOD]["track"]["RPrec"] = str(
            "{:10.3f}".format(np.mean(r_prec_list))) + " (" + str("{:10.3f}".format(scipy.stats.sem(r_prec_list))) + ")"
        METRIC_DICT[city_to_test][METHOD]["track"]["NDCG"] = str(
            "{:10.3f}".format(np.mean(ndcg_metric_list))) + " (" + str(
            "{:10.3f}".format(scipy.stats.sem(ndcg_metric_list))) + ")"
        METRIC_DICT[city_to_test][METHOD]["track"]["Prec@1"] = str(
            "{:10.3f}".format(np.mean(track_prec_at_one_list))) + " (" + str(
            "{:10.3f}".format(scipy.stats.sem(track_prec_at_one_list))) + ")"

        METRIC_DICT[city_to_test][METHOD]["track"]["RPrecVAL"] = np.mean(r_prec_list)
        METRIC_DICT[city_to_test][METHOD]["track"]["NDCGVAL"] = np.mean(ndcg_metric_list)
        METRIC_DICT[city_to_test][METHOD]["track"]["Prec@1VAL"] = np.mean(track_prec_at_one_list)
output_latex()


