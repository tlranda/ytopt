from math import comb, factorial
import pdb

def hyper(pop, cand, samp, req, big=False):
    if not big:
        t1 = comb(cand, req)*comb(pop-cand, samp-req)
        t2 = comb(pop, samp)
        return t1/t2
    else:
        accumulate = 0.
        for n_targets in range(1,cand+1):
            t1 = comb(cand,n_targets)*comb(pop-cand, samp-n_targets)
            t2 = comb(pop, samp)
            print(f"Hyper iteration {n_targets} = {t1/t2}")
            accumulate += (t1/t2)
        return accumulate

def get_contraction(pop, delta_pop, cand, delta_cand, samp, req, big=False):
    print("BEFORE")
    before = hyper(pop, cand, samp, req, big)
    print("AFTER")
    after  = hyper(pop+delta_pop, cand+delta_cand, samp, req, big)
    contraction = after / before
    return before, after, contraction

def delta_comb(a, delta_a, b):
    t1 = factorial(a+delta_a) / factorial(a)
    t2 = factorial(a-b) / factorial(a+delta_a-b)
    change = t1 * t2
    return change

def predict_ratio(pop, delta_pop, cand, delta_cand, samp, req, big=False):
    if not big:
        t1 = delta_comb(cand, delta_cand, req)
        t2 = delta_comb(pop-cand, delta_pop-delta_cand, samp-req)
        t3 = delta_comb(pop, delta_pop, samp)
        ratio = (t1 * t2) / t3
        return ratio
    else:
        accumulate = 0.
        for n_targets in range(1,cand+delta_cand+1):
            t1 = delta_comb(cand, delta_cand, n_targets)
            t2 = delta_comb(pop-cand, delta_pop-delta_cand, samp-n_targets)
            t3 = delta_comb(pop, delta_pop, samp)
            ratio = (t1 * t2) / t3
            print(f"PREDICT ITERATION {n_targets} = {ratio}")
            accumulate += ratio
        return accumulate

def nice_print(d, keys = None):
    if keys is None:
        keys = d.keys()
    longest = max(map(len,d.keys()))
    print("\n".join([f"{k:<{longest}}: {d[k]}" for k in keys])+"\n")

def simple_test():
    pop = 12
    delta_pop = -2
    cand = 3
    delta_cand = -1
    samp = 5
    req = 1
    big = True
    input_dict = dict((k,v) for (k,v) in locals().items())
    nice_print(input_dict)
    before, after, contraction = get_contraction(pop, delta_pop,
                                                 cand, delta_cand,
                                                 samp, req, big)
    #pdb.set_trace()
    ratio = predict_ratio(pop, delta_pop, cand, delta_cand, samp, req, big)
    local_dict = dict((k,v) for (k,v) in locals().items() if '_dict' not in k)
    new_results = sorted(set(local_dict.keys()).difference(set(input_dict.keys())))
    nice_print(local_dict, new_results)

if __name__ == '__main__':
    simple_test()

