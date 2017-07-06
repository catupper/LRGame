import numpy as np
from matplotlib import pyplot as plot
from scipy.stats import norm
import os
import random

EPS = 1e-10

class Data:
    def __init__(self, cnt, contStep, contdB, lum, TF, real, answer, sec, time):
        self.cnt = cnt
        self.contStep = contStep
        self.contdB = contdB
        self.lum = lum
        self.TF = TF
        self.real = real
        self.answer = answer
        self.sec = sec
        self.time = time

    def __repr__(self):
        return "%s %s %s %s %s %s %s %s %s"%(self.cnt, self.contStep, self.contdB, self.lum, self.TF, self.real, self.answer, self.sec, self.time)

def parse(str):
    if(len(str.strip()) == 0):return 
    cnt, contStep, contdB, lum, TF, real, answer, sec, time =  map(lambda x:x.strip(), str.split())
    cnt = int(cnt)
    contStep = int(contStep)
    contdB = float(contdB)
    lum = int(lum)
    TF = bool(int(TF))
    sec = float(sec)
    time = float(time)
    return Data(cnt, contStep, contdB, lum, TF, real, answer, sec, time)
    

def read_file(filename):
    with open(filename, "r") as f:
        data = [x for x in map(parse, f.read().split("\n")) if x != None]
        
        return data


def main():
    plot.subplot(131)
    plt(lambda x:"Red" in x)
    plot.subplot(132)
    plt(lambda x:"Green" in x)
    plot.subplot(133)
    plt(lambda x:"Blue" in x)

    plot.show()
    exit()
    
    plot.subplot(331)
    plt(lambda x:"Red" in x and "Ogura" in x)
    plot.subplot(332)
    plt(lambda x:"Green" in x and "Ogura" in x)
    plot.subplot(333)
    plt(lambda x:"Blue" in x and "Ogura" in x)
    
    plot.subplot(334)
    plt(lambda x:"Red" in x and "Castle" in x)
    plot.subplot(335)
    plt(lambda x:"Green" in x and "Castle" in x)
    plot.subplot(336)
    plt(lambda x:"Blue" in x and "Castle" in x)

    
    plot.subplot(337)
    plt(lambda x:"Red" in x and "Hatanaka" in x)
    plot.subplot(338)
    plt(lambda x:"Green" in x and "Hatanaka" in x)
    plot.subplot(339)
    plt(lambda x:"Blue" in x and "Hatanaka" in x)

    plot.show()
       
def plt(func):
    data = []
    for f in os.listdir("data2"):
        if func(f):
            data += read_file("data2/%s"%f)
    n, N, xs = conv(data)[1:]
    ave, var = 5.5, 0.5
    ave,var = estimate(data)
    print ave,var
    x = np.arange(0,10,0.01)
    plot.plot(x, 1 - (1 - norm.cdf(x, ave,var)) / 2)
    y = [1.0 * n[i] / N[i] for i in range(len(xs))]
    plot.plot(xs,y)

def deriv(f, x):
    dig = x.shape[0]
    res = []
    for i in range(dig):
        d = np.zeros(dig)
        d[i] += EPS
        res.append((f(x+d)-f(x)) / EPS)
    return np.array(res)

def jump(f, x, d):
    bottom, top = 0, 0.01
    for i in range(10):
        m1 = (bottom + bottom + top) / 3
        m2 = (bottom + top + top) / 3
        if(f(x + d*m1) > f(x+d*m2)):top = m2
        else:bottom = m1
    return bottom

def find_max(f, x):
    for i in range(300):
        d = deriv(f, x)
        j = jump(f,x,d)
        if j < EPS:break
        x = x + d * j
    return x


def conv(data):
    steps = list({x.contStep for x in data})
    steps.sort()
    n = []
    N = []
    for i in steps:
        n.append(0)
        N.append(0)
        for x in data:
            if x.contStep == i:
                N[-1] += 1
            if x.contStep == i and x.TF:
                n[-1] += 1
    return len(n), n, N, steps


def lpgen(data):
    def lp(x):
        ave, var = x
        p = lambda x: norm.cdf(x, ave,var)
        res = 0
        a, n, N, x = conv(data)
        for i in range(a):
            res += bilog(N[i], n[i]) + n[i] * np.log(0.5 + p(x[i]) * 0.5) + (N[i] - n[i]) * np.log((1 - p(x[i]) ) *0.5)
        return res
    return lp

def starling(x):
    if x == 0:return 0
    return np.log(np.sqrt(2*np.pi/x))  + x * np.log(x) - x
                
                
def bilog(n, k):
    return starling(n) - starling(k)  - starling(n-k)

def estimate(data):
    ave = 5.5
    var = 1.0
    lp = lpgen(data)
    ave, var = find_max(lp, np.array([ave, var]))
    return ave, var
        

if __name__ == "__main__":
    main()
    


