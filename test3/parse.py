import collections
import zlib
import functools
import os
import pickle

import time
import random
random.seed(time.time())

filename_usefulness = 'test3data/usefulness_raw.pickle'
filename_problemslemmas_test = 'test3data/problemslemmas_test_raw.pickle'
filename_problemslemmas_validation = 'test3data/problemslemmas_validation_raw.pickle'

def get_data():
    if (os.path.isfile(filename_usefulness)):
        usefulness = pickle.load(open(filename_usefulness, 'rb'))
    else:
        usefulness = _get_usefulness()
        os.makedirs(os.path.dirname(filename_usefulness))
        pickle.dump(usefulness, open(filename_usefulness, 'wb'))

    if (os.path.isfile(filename_problemslemmas_test) and os.path.isfile(filename_problemslemmas_validation)):
        problemslemmas_test = pickle.load(open(filename_problemslemmas_test, 'rb'))
        problemslemmas_validation = pickle.load(open(filename_problemslemmas_validation, 'rb'))
    else:
        problemslemmas_test, problemslemmas_validation = _get_problemslemmas()
        pickle.dump(problemslemmas_test, open(filename_problemslemmas_test, 'wb'))
        pickle.dump(problemslemmas_validation, open(filename_problemslemmas_validation, 'wb'))

    return usefulness, problemslemmas_test, problemslemmas_validation

@functools.lru_cache(maxsize=1)
def parse_problem(problemname):
    return _parse_cnf_file('../E_conj/problems/{}'.format(problemname))


# all private methods from here on =================================================================


def _parse_cnf_list(s):
    # Filter out comments
    s = '\n'.join(l for l in s.split('\n') if not l.startswith('#') and l)
    return s

def _parse_cnf_file(filename):
    with open(filename, 'r') as f:
        return _parse_cnf_list(f.read())

def _get_usefulness():
    print('getting usefulness')
    linecount = 0
    with open('../E_conj/statistics', 'r') as f:
        s = f.read()
        ls = s.split('\n')
        usefulness = collections.defaultdict(dict)
        for l in ls:
            linecount = linecount + 1
            if not l.strip():
                continue
            psr, problem, lemmaname, *_ = l.split(':')
            psr = float(psr)
            lemmaname = lemmaname.split('.')[0]
            usefulness[problem][lemmaname] = psr

    print('statistics lines parsed: ', linecount)
    return usefulness

def _process_problemslemmas(l):
    name, lemma = l.split(':')
    _, problemname, lemmaname = name.split('/')
    return (
        problemname,
        lemmaname,
        parse_problem(problemname),
        lemma,
        )

def _get_problemslemmas():
    print('parsing problems and lemmas')
    with open('../E_conj/lemmas', 'r') as f:
        s = f.read()
        ls = s.split('\n')
        problemslemmas_test = list()
        problemslemmas_validation = list()
        for i in range(0, len(ls) - 1):
            processed_problemlemma = _process_problemslemmas(ls[i])
            if (random.random() <= 0.8):
                problemslemmas_test.append(processed_problemlemma)
            else:
                problemslemmas_validation.append(processed_problemlemma)

    print('problemlemma test lines parsed: ', len(problemslemmas_test))
    print('problemlemma validation lines parsed: ', len(problemslemmas_validation))
    return problemslemmas_test, problemslemmas_validation
    
# Code below ran into trouble on windows - infinite loops of threads called
'''
def _get_problemslemmas():
    print('parsing problems and lemmas')
    import multiprocessing

    with multiprocessing.Pool() as pool:
        with open('E_conj/lemmas') as f:
            return pool.map(_process_problemslemmas, f, 32)
'''
