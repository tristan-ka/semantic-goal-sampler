import itertools

def variations2permutations(variations_dict):
    keys, values = zip(*variations_dict.items())
    return [dict(zip(keys,v)) for v in itertools.product(*values)]