# import numpy as np
from sklearn.ensemble import RandomForestClassifier

domainlist = []


class Domain:
    def __init__(self, _name, _label, _len, _numbers):
        self.name = _name
        self.label = _label
        self.len = _len
        self.numbers = _numbers

    def returnData(self):
        return [self.len, self.numbers]

    def returnLabel(self):
        if self.label == "notdga":
            return 0
        else:
            return 1


def findNum(str):
    counter = 0
    for i in str:
        if i.isdigit():
            counter += 1
    return counter


def initData(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            tokens = line.split(",")
            name = tokens[0]
            label = tokens[1]
            _len = len(tokens[0])
            numbers = findNum(tokens[0])
            domainlist.append(Domain(name, label, _len, numbers))


def main():
    initData("train.txt")
    featureMatrix = []
    labelList = []
    for item in domainlist:
        featureMatrix.append(item.returnData())
        labelList.append(item.returnLabel())

    clf = RandomForestClassifier(random_state=0)
    clf.fit(featureMatrix, labelList)
    with open('test.txt') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            _len = len(line)
            _numbers = findNum(line)
            with open('result.txt', 'a') as _f:
                if clf.predict([[_len, _numbers]]):
                    tempstr = [line, ", dga\n"]
                    _f.writelines(tempstr)
                else:
                    tempstr = [line, ", notdga\n"]
                    _f.writelines(tempstr)


if __name__ == '__main__':
    main()
