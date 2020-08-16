"""
计算accuracy, precision, recall, f1-score
"""


def ner_accuracy(tag_pred_y, tag_true_y):
    acc = 0.0
    num = 0.0
    for j in range(len(tag_pred_y)):
        for z in range(len(tag_pred_y[j])):
            if tag_pred_y[j][z] == tag_true_y[j][z]:
                acc += 1
            num += 1
    return acc / num


def F1(y_pred, y_true):
    c = 0
    true = 0
    pos = 0
    for i in range(len(y_true)):
        start = 0
        for j in range(len(y_true[i])):
            if y_pred[i][j][0] == 'L' or y_pred[i][j][0] == 'S':
                pos += 1
            if y_true[i][j][0] == 'L' or y_true[i][j][0] == 'S':
                flag = True
                if y_pred[i][j] != y_true[i][j]:
                    flag = False
                if flag:
                    for k in range(start, j):
                        if y_pred[i][k] != y_true[i][k]:
                            flag = False
                            break
                    if flag:
                        c += 1
                true += 1
                start = i + 1
    p = c / float(pos)
    r = c / float(true)
    f = 2*p*r / (p+r)
    return p, r, f