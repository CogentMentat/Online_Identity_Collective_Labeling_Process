# coding: utf-8
"""
Run semiotic tagging models.  This is a "manual" version of multiprocessing,
splitting calculations based on input since numba jitclasses don't play nicely
with the multiprocessing module.

Designed to work with tagging classes in:

`semiotic_tagging_models.py`

Author: Alexander TJ Barron
Date Created: 2019-11-03

"""

import argparse
import os
import pickle
import logging

from time import time
from datetime import datetime

from semiotic_tagging_models import parameter_lattice_search

def main(simdirpath):

    logging.basicConfig(filename='{}_search.log'.format(simdirpath),
                        level=logging.INFO)

    # Get primetags to simulate.
    primetagslistpath = os.path.join(simdirpath, 'primetagslist.txt')
    with open(primetagslistpath, mode='r') as f:
        primetagslist = [line.strip() for line in f.readlines()]

    # Get simulation ingredients.
    d_tag_simingreds = {}
    for tag in primetagslist:
        tag_simingredspath = os.path.join(simdirpath,
                                          'simingreds_{}.pkl'.format(tag))
        with open(tag_simingredspath, mode='rb') as f:
            d_tag_simingreds[tag] = pickle.load(f)

    t0 = time()
    logging.info('Starting: {}'.format(datetime.fromtimestamp(t0)))

    for pk, primetag in enumerate(primetagslist):
        part_tf = time()
        logging.info('{} of {} primetags: '.format(pk, len(primetagslist)) + \
                '{:.2f} min'.format((part_tf - t0)/60.))

        # Change primetag, keep the rest of the args.
        tag_simingreds = d_tag_simingreds[tag]
        tag_simingreds['primetag'] = primetag
        tag_simingreds['simdirpath'] = simdirpath
        parameter_lattice_search(**tag_simingreds)

    tf = time()
    logging.info('Total time: {:.2f} min.'.format((tf - t0)/60.))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('simdirpath', type=str,
                        help='Path to directory containing simulation ingredients.')

    args = parser.parse_args()

    main(args.simdirpath)
