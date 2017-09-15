import sys

filename = sys.argv[1]
f = open(filename, 'r')

aStr = f.read()
aList = aStr.split()
aDict = dict()

for word in aList:
    if word in aDict:
        aDict[word] += 1
    else:
        aDict[word] = 1


f = open('Q1.txt', 'w')

cnt = 0
for key, value in aDict.items():
    f.write(str(key) + ' ' + str(cnt) + ' ' + str(value) + '\n')
    cnt += 1





