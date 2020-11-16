#
# Classification impurity split criteria ---- 'gini' & 'twoing'
#
#.....Importing all the packages...........
#
#
from collections import Counter
import numpy as np
#
#
#
def gini(left, right):
    sum=0
    if (len(left)+len(right)) == 0:
        return np.inf
    else:
        p = len(left) / (len(left)+len(right))
        counter = Counter(left)
        freq = np.array([v / len(left) for v in counter.values()])
        sum += p * (1 - np.sum(np.square(freq)))
        counter = Counter(right)
        freq = np.array([v / len(right) for v in counter.values()])
        sum += (1-p) * (1 - np.sum(np.square(freq)))
        return sum


def twoing(left_label, right_label):
    sum = 0
    huge_val = np.inf
    left_len, right_len, n = len(left_label), len(right_label), (len(left_label) + len(right_label))
    if n == 0:
        return np.inf
    else:
        labels = list(left_label) + list(right_label)
        n_classes = np.unique(labels)
        if (left_len != 0 & right_len != 0):
            for i in n_classes:
                idx = np.where(left_label == i)[0]
                li = (len(idx) / left_len)
                idx = np.where(right_label == i)[0]
                ri = (len(idx) / right_len)
                sum += (np.abs(li - ri))
            twoing_value = ((left_len / n) * (right_len / n) * np.square(sum)) / 4

        elif (left_len == 0):
            for i in n_classes:
                idx = np.where(right_label == i)[0]
                ri = (len(idx) / right_len)
                sum += ri
            twoing_value = ((left_len / n) * (right_len / n) * np.square(sum)) / 4

        else:
            for i in n_classes:
                idx = np.where(left_label == i)[0]
                li = (len(idx) / left_len)
                sum += li
            twoing_value = ((left_len / n) * (right_len / n) * np.square(sum)) / 4
        if twoing_value == 0:
            return (huge_val)
        else:
            return (1 / twoing_value)


