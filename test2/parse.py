import collections
import zlib
import functools
import os
import pickle

filename_usefulness = 'test2data/usefulness_raw.pickle'
filename_problemslemmas = 'test2data/problemslemmas_raw.pickle'

def get_usefulness_problemslemmas():
    if (os.path.isfile(filename_usefulness)):
        usefulness = pickle.load(open(filename_usefulness, 'rb'))
    else:
        usefulness = _get_usefulness()
        os.makedirs(os.path.dirname(filename_usefulness))
        pickle.dump(usefulness, open(filename_usefulness, 'wb'))

    if (os.path.isfile(filename_problemslemmas)):
        problemslemmas = pickle.load(open(filename_problemslemmas, 'rb'))
    else:
        problemslemmas = _get_problemslemmas()
        pickle.dump(problemslemmas, open(filename_problemslemmas, 'wb'))

    return usefulness, problemslemmas

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
    with open('../E_conj/statistics', 'r') as f:
        s = f.read()
        ls = s.split('\n')
        usefulness = collections.defaultdict(dict)
        for l in ls:
            if not l.strip():
                continue
            psr, problem, lemmaname, *_ = l.split(':')
            psr = float(psr)
            lemmaname = lemmaname.split('.')[0]
            usefulness[problem][lemmaname] = psr

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
        problemslemmas = list()
        for i in range(0, len(ls) - 1):
            processed_problemlemma = _process_problemslemmas(ls[i])
            problemslemmas.append(processed_problemlemma)
        
    return problemslemmas
    
# Code below ran into trouble on windows - infinite loops of threads called
'''
def _get_problemslemmas():
    print('parsing problems and lemmas')
    import multiprocessing

    with multiprocessing.Pool() as pool:
        with open('E_conj/lemmas') as f:
            return pool.map(_process_problemslemmas, f, 32)
'''
