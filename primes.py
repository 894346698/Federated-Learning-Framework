import random
import sys

def ipow(a, b, n):
    """calculates (a**b) % n via binary exponentiation, yielding itermediate
       results as Rabin-Miller requires"""
    A = a = int(a % n)
    yield A
    t=1
    #t = 1L
    #python 3 没有长整型
    while t <= b:
        t <<= 1

    # t = 2**k, and t > b
    t >>= 2
    
    while t:
        A = (A * A) % n
        if t & b:
            A = (A * a) % n
        yield A
        t >>= 1

def rabin_miller_witness(test, possible):
    """Using Rabin-Miller witness test, will return True if possible is
       definitely not prime (composite), False if it may be prime."""    
    return 1 not in ipow(test, possible-1, possible)

smallprimes = (2,3,5,7,11,13,17,19,23,29,31,37,41,43,
               47,53,59,61,67,71,73,79,83,89,97)

def default_k(bits):
    return max(40, 2 * bits)

def is_probably_prime(possible, k=None):
    if possible == 1:
        return True
    if k is None:
        k = default_k(possible.bit_length())
    for i in smallprimes:
        if possible == i:
            return True
        if possible % i == 0:
            return False
    #for i in xrange(k):
    # print(type(i))
    #print(type(k))
    for i in range(int(k)):  #python 3 没有xrange、没有long
        test = random.randrange(2, possible - 1) | 1
        if rabin_miller_witness(test, possible):
            return False
    return True

def generate_prime(bits, k=None):
    """Will generate an integer of b bits that is probably prime 
       (after k trials). Reasonably fast on current hardware for 
       values of up to around 512 bits."""    
    assert bits >= 8

    if k is None:
        k = default_k(bits)

    while True:
        possible = random.randrange(2 ** (bits-1) + 1, 2 ** bits) | 1
        if is_probably_prime(possible, k):
            return possible

