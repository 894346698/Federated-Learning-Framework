import math
from exp import exp_mode
import numpy as np
np.set_printoptions(suppress=True)
import primes

# -*- coding: UTF-8 -*-
def invmod(a, p, maxiter=1000000):
    """The multiplicitive inverse of a in the integers modulo p:
         a * b == 1 mod p
       Returns b.
       (http://code.activestate.com/recipes/576737-inverse-modulo-p/)"""
    if a == 0:
        raise ValueError('0 has no inverse mod %d' % p)
    r = a
    d = 1
    for i in range(min(p, maxiter)):
        d = ((p // r + 1) * d) % p
        r = (d * a) % p
        if r == 1:
            break
    else:
        raise ValueError('%d has no inverse mod %d' % (a, p))
    return d

def modpow(base, exponent, modulus):
    """Modular exponent:
         c = b ^ e mod m
       Returns c.
       (http://www.programmish.com/?p=34)"""
    exponent=int(exponent)
    result = 1
    while exponent > 0:
        if int(exponent) & 1 == 1:
            result = (result * base) % modulus
        exponent = exponent >> 1
        base = (base * base) % modulus
    return result

class PrivateKey(object):

    def __init__(self, p, q, n):
        self.l = (p-1) * (q-1)
        self.m = invmod(self.l, n)

    def __repr__(self):
        return '<PrivateKey: %s %s>' % (self.l, self.m)
class PublicKey(object):


    @classmethod
    def from_n(cls, n):
        return cls(n)

    def __init__(self, n):
        self.n = n
        self.n_sq = n * n
        self.g = n + 1

    def __repr__(self):
        return '<PublicKey: %s>' % self.n

def generate_keypair(bits):
    p = primes.generate_prime(bits / 2)
    q = primes.generate_prime(bits / 2)
    n = p * q
    return PrivateKey(p, q, n), PublicKey(n)

def encrypt(pub, plain):
    '''
    while True:
        r = primes.generate_prime(int(round(math.log(pub.n, 2))))   #python 3 没有long
        if r > 0 and r < pub.n:
            break
    '''
    r = 3
    x = pow(r, pub.n, pub.n_sq)
    cipher = (pow(pub.g, plain, pub.n_sq) * x) % pub.n_sq
    return cipher

def encrypt1(pub, plain):
    '''
    while True:
        r = primes.generate_prime(int(round(math.log(pub.n, 2))))   #python 3 没有long
        if r > 0 and r < pub.n:
            break
    '''
    if plain<0:
        plain=pub.n+plain
    r = 3
    x = pow(r, pub.n, pub.n_sq)
    cipher = (pow(pub.g, plain, pub.n_sq) * x) % pub.n_sq
    return cipher



def e_add(pub, a, b):
    """Add one encrypted integer to another"""

    return a * b % pub.n_sq

def e_add_const(pub, a, n):
    """Add constant n to an encrypted integer"""
    return a * modpow(pub.g, n, pub.n_sq) % pub.n_sq

def e_mul_const(pub, a, n):
    """Multiplies an ancrypted integer by a constant"""
    return modpow(a, n, pub.n_sq)

def decrypt(priv, pub, cipher):
    x = int(exp_mode(cipher, priv.l, pub.n_sq)) - int(1)
    plain = ((int(x) // pub.n) * priv.m) % pub.n
    #if(plain>pub.n/2):
        #plain=0
    return plain

def decrypt1(priv, pub, cipher):
    x = int(exp_mode(cipher, priv.l, pub.n_sq)) - int(1)
    plain = ((int(x) // pub.n) * priv.m) % pub.n
    if(plain>pub.n/2):
        plain=plain-pub.n
    return plain


def decrypt2(priv, pub, cipher):
    x = int(exp_mode(cipher, priv.l, pub.n_sq)) - int(1)
    plain = ((int(x) // pub.n) * priv.m) % pub.n
    if(plain>pub.n/2):
        plain=0
    return plain

def matrix_e_mul_const(pub,A,B):#A加密，B明文#32位，A,B矩阵乘
    #print("B",B)
    v = [[0 for i in range(len(B[0]))] for i in range(len(A))]
    for i in range(len(A)):
        for k in range(len(B[0])):
            #tol = encrypt(pub,int(0))
            for j in range(len(A[0])):
                if j==0:
                    e=e_mul_const(pub,A[i][j],B[j][k])
                    #print("A[i][j]=",decrypt(priv, pub,A[i][j]) /100000000,"int(B[j][k])",B[j][k])
                    #print("e",i,j,decrypt(priv, pub,e) /100000000)
                else:
                    e=int(e_add(pub,e_mul_const(pub,A[i][j],int(B[j][k])),e))
                    #print("e", i, j, decrypt(priv, pub, e) / 10000000000)

            v[i][k]=e
    mingwenv = [[0 for i in range(len(v[0]))] for i in range(len(v))]

    return v

def matrix_e_mul_const2(pub,A,B):#A加密，B明文#32位，A,B矩阵乘
    #print("B",B)
    v = [[0 for i in range(len(B[0]))] for i in range(len(A))]
    for i in range(len(A)):
        for k in range(len(B[0])):
            for j in range(len(A[0])):
                if j==0:
                    e=e_mul_const(pub,B[j][k],A[i][j])

                else:
                    e=int(e_add(pub,e_mul_const(pub,int(B[j][k]),A[i][j]),e))

            v[i][k]=e
    mingwenv = [[0 for i in range(len(v[0]))] for i in range(len(v))]

    return v





def matrix_sub(pub,A,B):#A加密，B加密A-B
    #print("matrix_inverse")
    #print(pub.n)
    v = [[0 for i in range(len(B[0]))] for i in range(len(B))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            #v[i][j]=v[i][j]**(pub.n-1)% pub.n_sq
            v[i][j]=e_add(pub,A[i][j],modpow(B[i][j],pub.n-1,pub.n_sq))
    #print("matrix_inverse_end")
    return v


def transpose(A):
    B=[[1] * len(A) for i in range(0, len(A[0]))]
    for i in range(len(A[0])):
        for j in range(len(A)):
            B[i][j]=A[j][i]
    return B

def point_multiplication(A,B):
    C= [[1] * len(A[0]) for i in range(0, len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            C[i][j]=A[i][j]*B[i][j]
    return C


def jiami_e_mul_const(pub,A,b):
    B = [[0 for i in range(len(A[0]))] for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            B[i][j] = e_mul_const(pub, A[i][j],b)

    return B

def is_equal(A,B):
    b=True
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j]==B[i][j]:continue
            else:
                b=False
                break;
    return b

def is_equal_0(A):
    b = True
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] == 0:
                continue
            else:
                b = False
                break;
    return b