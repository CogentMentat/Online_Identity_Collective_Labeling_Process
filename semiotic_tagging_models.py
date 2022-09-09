# coding: utf-8
"""
Simulation code for different semiotic tagging models.

Inspired from:

Cattuto, C., Loreto, V. & Pietronero, L. Semiotic dynamics and collaborative
tagging. PNAS 104, 1461–1464 (2007).

Author: Alexander TJ Barron
Date Created: 2019-11-01

"""

import pdb

import os
import sys
import pickle

import numpy as np

import numba

from time import time
from collections import Counter

spec = [
    ('T', numba.int32),
    ('p_param', numba.float32),
    ('tau_param', numba.int32),
    ('n0_param', numba.int32),
    ('umbrellakernel_unnorm', numba.float64[:]),
]
@numba.experimental.jitclass(spec)
class TagSimulator_pnew_unboundvoc:

    def __init__(self, T, p_param, tau_param, n0_param):

        self.T = T
        self.p_param = p_param
        self.tau_param = tau_param
        self.n0_param = n0_param

        # Pre-calculate some of any kernel.  (Full pre-calculation not available via dict,
        #                                     since dict isn't supported in @numba.jitclass.)
        backwards_kernel_unnorm = 1/(np.arange(1, T+1) + tau_param)
        self.umbrellakernel_unnorm = backwards_kernel_unnorm[::-1].astype(np.float64)

    def simulate(self, seed):

        np.random.seed(seed)

        # Pre-allocate history array
        history = np.zeros(self.T, dtype=np.uint32)

        # Set initial history.
        history[:self.n0_param] = np.arange(self.n0_param)

        # Build history.
        newtag = np.uint32(max(history) + 1)
        for t in range(self.n0_param, self.T):

            # Flip biased coin.
            flip = np.random.rand()
            if flip <= self.p_param:
                coin = True
            else:
                coin = False

            if coin: # Pick a new tag.

                history[t] = newtag
                newtag += 1

            else: # Pick an existing tag.
                kernel_unnorm = self.umbrellakernel_unnorm[-t:]
                kernel = kernel_unnorm/(kernel_unnorm.sum())
                # Workaround, since numba doesn't support the 'p' option in
                # numpy.random.choice
                idx = np.searchsorted(np.cumsum(kernel), np.random.random(), side="right")
                tag = history[:t][idx]
                history[t] = tag

        return history

spec = [
    ('T', numba.int32),
    ('p_param', numba.float32),
    ('tau_param', numba.int32),
    ('n0_param', numba.int32),
]
@numba.experimental.jitclass(spec)
class TagSimulator_pnew_unboundvoc_Yule:

    def __init__(self, T, p_param, tau_param, n0_param):

        self.T = T
        self.p_param = p_param
        self.tau_param = tau_param # unused in this class
        self.n0_param = n0_param

    def simulate(self, seed):

        np.random.seed(seed)

        # Pre-allocate history array
        history = np.zeros(self.T, dtype=np.uint32)

        # Set initial history.
        history[:self.n0_param] = np.arange(self.n0_param)

        # Build history.
        newtag = np.uint32(max(history) + 1)
        for t in range(self.n0_param, self.T):

            # Flip biased coin.
            flip = np.random.rand()
            if flip <= self.p_param:
                coin = True
            else:
                coin = False

            if coin: # Pick a new tag.

                history[t] = newtag
                newtag += 1

            else: # Pick an existing tag.
                idx = np.random.randint(t)
                tag = history[:t][idx]
                history[t] = tag

        return history

def get_simul_countsprobs_perrank(sim_tag_histories, transient=0):
    """
    Take a list of simulated tag history, get rank-freq probabilities.

    """

    simul_cnts = []
    simul_probs = []
    simul_ranks = []
    for sim_tag_history in sim_tag_histories:
        sim_tag_history = sim_tag_history[transient:]
        cooc = Counter(sim_tag_history)
        rank, count = zip(*enumerate(sorted(cooc.values(), reverse=True)))
        ranks = np.array(rank)+1
        probs = np.array(count)/sum(count)

        simul_cnts.append(count)
        simul_probs.append(probs)
        simul_ranks.append(ranks)

    max_rank = max(max(ranks) for ranks in simul_ranks)

    # counts
    simul_cnts_arrs = [np.array(list(counts) + [np.nan]*(max_rank-len(counts))) \
                        for counts in simul_cnts]
    simul_cnts_stack = np.vstack(simul_cnts_arrs)
    simul_cnts_perrank = simul_cnts_stack.T
    # Get rid of array - it was useful for the transpose only.
    simul_cnts_perrank = [row[~np.isnan(row)] for row in simul_cnts_perrank]

    # probs
    simul_probs_arrs = [np.array(list(probs) + [np.nan]*(max_rank-len(probs))) \
                        for probs in simul_probs]
    simul_probs_stack = np.vstack(simul_probs_arrs)
    simul_probs_perrank = simul_probs_stack.T
    # Get rid of array - it was useful for the transpose only.
    simul_probs_perrank = [row[~np.isnan(row)] for row in simul_probs_perrank]

    return simul_cnts_perrank, simul_probs_perrank

def score_simul_rankprobmeans(rankprobmeans, realrankprobs,
                                weighted=False, weightedscore=False):

    realrankprobs = list(realrankprobs)
    # Under certain parameter conditions, not all words are used by the iteration
    # count T.  This is the reason for appending 0s, below.
    if len(rankprobmeans) < len(realrankprobs):
        allrankprobmeans = rankprobmeans + [0]*(len(realrankprobs) - len(rankprobmeans))
        allrealrankprobs = realrankprobs
    elif len(rankprobmeans) > len(realrankprobs):
        allrealrankprobs = realrankprobs + [0]*(len(rankprobmeans) - len(realrankprobs))
        allrankprobmeans = rankprobmeans
    else:
        allrankprobmeans = rankprobmeans
        allrealrankprobs = realrankprobs
    if weightedscore:
        #weights = 1/np.arange(1, len(allrankprobmeans)+1)
        weights = len(allrankprobmeans) - -.5*np.arange(1, len(allrankprobmeans)+1)
        weights = weights/weights.sum()
        return sum([np.absolute(x[0] - x[1])*x[2] for x in zip(allrankprobmeans,
            allrealrankprobs, weights)])
    else:
        return sum([np.absolute(x[0] - x[1]) for x in zip(allrankprobmeans, allrealrankprobs)])

def score_simtaghistories(sim_tag_histories, real_rankprobs,
                            transient=0, weightedscore=False):
    simul_cnts_perrank, simul_probs_perrank = \
            get_simul_countsprobs_perrank(sim_tag_histories,
                                          transient=transient)
    simul_rankprobmeans = [np.mean(sp_pr) for sp_pr in simul_probs_perrank]
    score = score_simul_rankprobmeans(simul_rankprobmeans,
                              real_rankprobs,
                              weightedscore=weightedscore)
    return score

def parameter_lattice_search(taggingclassname=None, primetag=None,
        d_hashtag_cooc=None, tagprofiles=None, simcount=None,
        T=None, searchiternum=None, param_combos=None, transient=None,
        simdirpath=None, save_histories=False, weightedscore=False):
    """
    Search a parameter lattice.

    taggingclassname (str): class name running a particular semiotic tagging model
    primetag (str): Prime tag for modified Yule-Simon simulation.
    d_hashtag_cooc (dict): Dictionary mapping prime tags to sub-dictionaries:
      each mapping the co-occurring tags to their co-occurrence counts.
    tagprofiles (list): List of sets of tags which co-occur.
    simcount (int): Number of simulations per parameter combination.
    T (int): Number of coin flips.
    searchiternum (int): Number of searches to perform over all parameter combinations.
    param_combos (list): List of parameter combinations, each a dictionary with
      values for keys in ['p', 'tau', 'n0'].
    transient (int): Length of history to remove from the beginning of a simulation.
    simdirpath (str): Path of directory to save simulated tag histories.
    weightedscore (bool): whether to weight lower ranks higher,
      according to a power law with exponent -1

    """

    primetag_cooc = d_hashtag_cooc[primetag]

    # Real rank-freq probabilities.
    rank, realcounts = zip(*enumerate(sorted(primetag_cooc.values(), reverse=True)))
    realrankprobs = np.array(realcounts)/sum(realcounts)

    ## ==== Conduct [searchiternum] searches over all parameter combinations,
    ## simulating [simcount] models each time before obtaining the score.

    primetagsearchiter_simtaghistoryarrs = []
    for searchiter in range(searchiternum):

        # Get seeds for each parameter combination in the current search.
        # Don't simplify this to a list comprehension.  os.urandom doesn't
        # play nice, and will give the same integer for every iteration in
        # the comprehension.
        seeds_perparamcombo = []
        for param_iter in range(len(param_combos)):
            seeds = []
            for simul in range(simcount):
                seeds.append(int.from_bytes(os.urandom(4), byteorder='little'))
            seeds_perparamcombo.append(seeds)

        # Worker tuples.
        wtups = [(taggingclassname,
                     primetag,
                     primetag_cooc,
                     simcount,
                     paramcombo,
                     T,
                     transient,
                     realrankprobs,
                     seeds) for seeds, paramcombo in zip(seeds_perparamcombo, param_combos)]

        paramcombo_simoutsdicts = []
        d_paramcombo_simtaghistories = {}
        for paramcombo_k, wtup in enumerate(wtups):

            taggingclassname, \
            primetag, \
            primeht_cooc, \
            simcount, \
            paramcombo, \
            T, \
            transient, \
            real_rankprobs, \
            seeds = wtup

            n0_param = paramcombo['n0']
            p_param = paramcombo['p']
            tau_param = paramcombo['tau']
            primeht_cooc_hts = list(primetag_cooc.keys())
            # Get tagging simulation class.
            TagSim = eval("{}".format(taggingclassname) + \
                    "(T, p_param, tau_param, n0_param)")

            # Record simulated tag histories.
            sim_tag_histories = []
            for seed in seeds:
                sim_tag_histories.append(TagSim.simulate(seed))

            score = score_simtaghistories(sim_tag_histories,
                                          realrankprobs,
                                          transient=transient,
                                          weightedscore=weightedscore)

            # Get param combo label/key to link the simulation out dict
            # to the .npz file array name.
            paramcombo_key = 'paramcombo_{}'.format(paramcombo_k)

            # Store param combo simulation out dict.
            paramcombo_simoutsdict = {'primetag': primetag,
                            'paramcombo': paramcombo,
                            'paramcombo_key': paramcombo_key,
                            'score': score,
                            'transient': transient}
            paramcombo_simoutsdicts.append(paramcombo_simoutsdict)

            # Store param combo simulation out dict.
            d_paramcombo_simtaghistories[paramcombo_key] = \
                    np.vstack(sim_tag_histories)


        # Save the current batch of simulated tag histories, one array of
        # simcount tag histories per param combo, for this search iteration.
        if save_histories:
            outname = "{}_searchiter{}_paramcombosimtaghistories.npz".format(primetag,
                                                                             searchiter)
            outpath = os.path.join(simdirpath, outname)
            with open(outpath, mode='wb') as f:
                np.savez(f, **d_paramcombo_simtaghistories)

        # Save the current batch of simulation results with this
        # parameter combo.
        outname = "{}_searchiter{}_paramcombosimoutsdicts.pkl".format(primetag,
                                                                      searchiter)
        outpath = os.path.join(simdirpath, outname)
        with open(outpath, mode='wb') as f:
            pickle.dump(paramcombo_simoutsdicts, f)

