import collections
import zlib
import functools
import os
import pickle
import parsy as p

import time
import random
random.seed(time.time())

filename_usefulness = 'test4data/usefulness_raw.pickle'
filename_problemslemmas_test = 'test4data/problemslemmas_test_raw.pickle'
filename_problemslemmas_validation = 'test4data/problemslemmas_validation_raw.pickle'
filename_mappings = 'mappings.pickle'

# Retrieve mappings for serialization
# - Const: map 1-335
# - Func: map 1-5010
# - Disj name: map 1-3332
# - Disj role: 1 for axiom, 2 for negated_conjecture, 3 for plain
# - Var --> 1-49 for X1-X49
# - Dist --> 0-64
# - Eq --> 1 for True, 0 for False
mappings = pickle.load(open(filename_mappings, 'rb'))
const_mapping = mappings['const_mapping']
func_mapping = mappings['func_mapping']
disj_name_mapping = mappings['disj_name_mapping']
disj_role_mapping = {'axiom': 1, 'negated_conjecture': 2, 'plain': 3}
var_mapping = lambda var_name: int(var_name[1:])
dist_mapping = lambda dist_name: int(dist_name)
eq_mapping = lambda eq_pos: 1 if eq_pos else 0

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
    return [serialize(token) for token in cnf_list.parse(s)]

def _parse_cnf_clause(s):
    return [serialize(token) for token in cnf_annotated.parse(s)]

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
    name, lemma = l.split(': ')
    _, problemname, lemmaname = name.split('/')
    return (
        problemname,
        lemmaname,
        parse_problem(problemname),
        _parse_cnf_clause(lemma),
        )

def _get_problemslemmas():
    print('parsing problems and lemmas')
    with open('../E_conj/lemmas', 'r') as f:
        s = f.read()
        ls = s.split('\n')
        problemslemmas_test = list()
        problemslemmas_validation = list()
        for i in range(0, len(ls) - 1):
            if i % 1000 == 0:
                print('[%d/%d] parsed' % (i, len(ls)))
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

# ==============================================
# Tokenizer methods

# Serializes the token, e.g. 'Func k5_finsub_1' --> 'Func 2408', 'Disj i_0_8 plain' --> 'Disj 3111 3'
def serialize(token):
    if token in ['Conj', '(', ')']:
        return token
    splitted = token.split()
    keyword = splitted[0]
    if keyword == 'Var':
        return 'Var %d' % var_mapping(splitted[1])
    elif keyword == 'Dist':
        return 'Dist %d' % dist_mapping(splitted[1])
    elif keyword == 'Const':
        return 'Const %d' % const_mapping[splitted[1]]
    elif keyword == 'Func':
        return 'Func %d' % func_mapping[splitted[1]]
    elif keyword == 'Eq':
        return 'Eq %d' % eq_mapping(splitted[1])
    elif keyword == 'Disj':
        return 'Disj %d %d' % (disj_name_mapping[splitted[1]], disj_role_mapping[splitted[2]])
    else:
        raise Exception('Unrecognized token %s' % token)

def flatten(ls):
    return [item for sublist in ls for item in sublist]

def symbol(s):
    return p.whitespace.many() >> p.string(s) << p.whitespace.many()

def parenthesised(parser):
    @p.generate
    def a():
        # Handle parenthesised terms
        if (yield symbol('(').optional()):
            t = yield a
            yield symbol(')')
            return t
        else:
            return (yield parser)
    return a

alpha_numeric = p.regex('[a-zA-Z0-9_]')

lower_alpha = p.regex('[a-z]')
upper_alpha = p.regex('[A-Z]')

lower_word = p.seq(lower_alpha, alpha_numeric.many().concat()).concat()
upper_word = p.seq(upper_alpha, alpha_numeric.many().concat()).concat()

integer = p.seq(
            p.string_from('+', '-').optional(),
            p.decimal_digit.at_least(1).concat().map(int)
        ).combine(lambda sign, number: -number if sign == '-' else number)

sq_char = p.regex('[a-zA-Z0-9 _\\-/~!@#$%^&*(),."]')
single_quote = p.string("'")
single_quoted = single_quote >> sq_char.at_least(1).concat() << single_quote

dq_char = p.regex("[a-zA-Z0-9 _\\-/~!@#$%^&*(),.']")
double_quote = p.string('"')
double_quoted = double_quote >> dq_char.at_least(1).concat() << double_quote

atomic_word = lower_word | single_quoted

name = atomic_word | integer

formula_role = p.string_from(
    'axiom',
    'hypothesis',
    'definition',
    'assumption',
    'lemma',
    'theorem',
    'corollary',
    'conjecture',
    'negated_conjecture',
    'plain',
    'type',
    'fi_domain',
    'fi_functors',
    'fi_predicates',
    'unknown')

constant = (lower_word | p.string('$') + lower_word | p.string('$$') + lower_word).map(lambda x: 'Const ' + x)
variable = upper_word.map(lambda x: 'Var ' + x)
distinct = (integer.map(repr) | double_quoted.map(repr)).map(lambda x: 'Dist ' + x)

@p.generate
def fof_term():
    v = yield variable.optional()
    if v:
        return [v]

    c = yield constant | distinct

    if not (yield symbol('(').optional()):
        return [c]

    args = yield function_args
    yield symbol(')')
    return ['Func %s' % c.replace('Const ', ''), '(', *args, ')']

fof_term = parenthesised(fof_term)

function_args = fof_term.sep_by(symbol(',')).map(flatten)

@p.generate
def literal():
    n = yield symbol('~').optional()
    t = yield fof_term
    eq = yield (symbol('=')|symbol('!=')).optional()
    if eq is None:
        if n:
            return ['Eq False', '(', *t, ')', 'Const $true']
        else:
            return ['Eq True', '(', *t, ')', 'Const $true']
    else:
        pos = n == (eq == '!=')
        t2 = yield fof_term
        return ['Eq %s' % pos, '(', *t, ')', '(', *t2]
literal = parenthesised(literal)

disjunction = literal.sep_by(symbol('|')).map(flatten)

cnf_formula = parenthesised(disjunction)

@p.generate
def cnf_annotated():
    yield symbol('cnf')
    yield symbol('(')
    n = yield name
    yield symbol(',')
    fr = yield formula_role
    yield symbol(',')
    cf = yield cnf_formula
    yield symbol(')')
    yield symbol('.')
    return ['Disj %s %s' % (n, fr), '(', *cf, ')']

cnf_list = cnf_annotated.many().map(flatten).map(lambda x: ['Conj', '(', *x, ')'])

if __name__ == '__main__':
    test1 = 'cnf(i_0_23, plain, (v2_struct_0(X1)|v2_collsp(X1)|~l1_collsp(X1)|~r1_collsp(X1,esk9_1(X1),esk9_1(X1),esk10_1(X1))|~r1_collsp(X1,esk9_1(X1),esk10_1(X1),esk9_1(X1))|~r1_collsp(X1,esk9_1(X1),esk10_1(X1),esk10_1(X1)))). cnf(i_0_8, plain, (v4_finsub_1(k5_finsub_1(X1)))).'
    parsed1 = cnf_list.parse(test1)
