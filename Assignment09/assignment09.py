import numpy as np
import collections



# a = np.array(['0', '3', '0', '1', '0', '1', '2', '1', '0', '0', '0', '0', '1', '3', '4'])

# pprint(collections.Counter(a)['0'])
# pprint(collections.Counter(a))


file_data		= "mnist_train.csv"
handle_file	= open(file_data, "r")
data        		= handle_file.readlines()
handle_file.close()

size_row	= 28    # height of the image
size_col  	= 28    # width of the image

num_image	= len(data)
count       	= 0     # count for the number of images


def normalize(data):
    data_normalized = (data - min(data)) / (max(data) - min(data))
    return(data_normalized)


list_label  = np.empty(num_image, dtype=int)

zero_data = []
zero_data_y = []
for line in data:
	
	line_data   = line.split(',')
	label       = line_data[0]
	im_vector   = np.asfarray(line_data[1:])
	im_vector   = normalize(im_vector)
	im_vector = np.insert(im_vector, 0, 1)
	zero_data.append(im_vector)
	list_label[count]       = label
	if label == '0':
		zero_data_y.append(1.0);
	else:
		zero_data_y.append(-1.0);
	# 1을 추가해서 A만들고
	count += 1


xn = np.array(zero_data,dtype=float)
yn = np.array(zero_data_y,dtype=float)
X = np.dot(np.linalg.pinv(xn) , yn)
truthZeroCount = collections.Counter(list_label)[0]
truthNotZeroCount = len(list_label) - truthZeroCount
answerZeroCount_y = 0
answerZeroCount_n = 0
answerNotZeroCount_y = 0
answerNotZeroCount_n = 0


for x in range(len(xn)):
	value = np.dot(xn[x],X)
	if value >= 0.0:
		if list_label[x] == 0:
			answerZeroCount_y = answerZeroCount_y + 1
		else:
			answerZeroCount_n = answerZeroCount_n + 1
	else:
		if list_label[x] != 0:
			answerNotZeroCount_y = answerNotZeroCount_y + 1
		else:
			answerNotZeroCount_n = answerNotZeroCount_n + 1
		# print('not zero' ,list_label[x] )

print(answerZeroCount_y,answerZeroCount_n,answerNotZeroCount_y,answerNotZeroCount_n)

TP = answerZeroCount_y/truthZeroCount
FP = answerNotZeroCount_n/truthNotZeroCount
FN = answerZeroCount_n/truthZeroCount
TN = answerNotZeroCount_y/truthNotZeroCount

pprint("TP : " + str(TP))
pprint("FP : " + str(FP))
pprint("FN : " + str(FN))
pprint("TN : " + str(TN))


