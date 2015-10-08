import os
from glob import glob

PATH = '/datasets/LivDet2015/Training/GreenBit'
result = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.*'))]
out = []
for line in result:
	if 'LIVE' in line.upper():
		out.append(line+" 1")
	else:
		out.append(line+" 0")

#print out
with open("greenbit.txt", 'w') as file:
	file.write("\n".join(out))
