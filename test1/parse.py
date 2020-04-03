import collections
import zlib
import functools

def parse_cnf_list(s):
    # Filter out comments
    s = '\n'.join(l for l in s.split('\n') if not l.startswith('#') and l)
    return s

def parse_cnf_file(filename):
    with open(filename, 'r') as f:
        return parse_cnf_list(f.read())

def get_usefulness():
    print('getting usefulness')
    with open('E_conj/statistics', 'r') as f:
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

@functools.lru_cache(maxsize=1)
def parse_problem(problemname):
    return parse_cnf_file('E_conj/problems/{}'.format(problemname))

def _process_problemslemmas(l):
    name, lemma = l.split(':')
    _, problemname, lemmaname = name.split('/')
    return (
        problemname,
        lemmaname,
        parse_problem(problemname),
        lemma,
        )

def get_problemslemmas():
    print('parsing problems and lemmas')
    import multiprocessing

    with multiprocessing.Pool() as pool:
        with open('E_conj/lemmas') as f:
            return pool.map(_process_problemslemmas, f, 32)