import sys

filename = sys.argv[1]
f = open(filename, 'r')

aStr = f.read()
aList = aStr.split()
bList = []

f = open('Q1.txt', 'w')

cnt = 0

for word in aList:
	if(word not in bList):
		bList.append(word)
		f.write(word + ' ' + str(cnt) + ' ' + str(aList.count(word)) + '\n')
		cnt += 1
		







