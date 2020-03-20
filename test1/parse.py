def parse_cnf_list(s):
    # Filter out comments
    s = '\n'.join(l for l in s.split('\n') if not l.startswith('#') and l)
    return s

def parse_cnf_file(filename):
    with open(filename, 'r') as f:
        return parse_cnf_list(f.read())
