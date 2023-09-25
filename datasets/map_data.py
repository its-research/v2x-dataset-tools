import torch
import numpy as np
import json
from typing import Callable, Dict, List, Optional, Tuple, Union


def convert(a:str):
    b = a.split(',')
    b = [float(b[0][1:]), float(b[1][:-1])]
    return b 

def is_range(ctr,x,y,len_range):
    if x-len_range<ctr[0]<x+len_range and y-len_range<ctr[1]<y+len_range:
        return True
    else:
        return False

def distance(x1,y1,x2,y2):
    dis = ((x1-x2)**2+(y1-y2)**2)**(0.5)
    return dis


def uniq(a:List[str]):
    for i in a :
        if a.count(i) > 1:
            a.remove(i)
    return a 

def convert2(a):
    ans = []
    for i in a:
        i = i.split(',')
        result = [float(i[0][1:]),float(i[1][:-1])]
        ans.append(result)
    return ans
