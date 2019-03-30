import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
from pprint import pprint
from collections import Counter



# a = [[1,2,3],[4,5,6],[7,8,9]]
# b = [1,1,1]

# a = np.array(a)
# b = np.array(b)
# print(a - b)


# exit()

file_data		= "mnist_train.csv"
handle_file	= open(file_data, "r")
data        		= handle_file.readlines()
handle_file.close()

size_row	= 28    # height of the image
size_col  	= 28    # width of the image

num_image	= len(data)
count       	= 0     # count for the number of images

#
# normalize the values of the input data to be [0, 1]
#
def normalize(data):

    data_normalized = (data - min(data)) / (max(data) - min(data))

    return(data_normalized)

#
# example of distance function between two vectors x and y
#
def distance(x, y):

    d = (x - y) ** 2
    s = np.sum(d)
    # r = np.sqrt(s)

    return(s)

#
# make a matrix each column of which represents an images in a vector form 
#
list_image  = np.empty((size_row * size_col, num_image), dtype=float)
list_label  = np.empty(num_image, dtype=int)


# list_image안에 있는 데이터 중 k개의 centroid를 random 으로 select 한다.
def init_centroid_select_random(count,k):
	global cluster_centroid
	global list_image
	for i in range(1 , k+1):
		cluster_centroid[i] = list_image[:, random.randrange(1,count)]


def init():
	for i in range(1,K+1):
		cluster[i] = []

	for line in data:
		line_data   = line.split(',')
		label       = line_data[0]
		im_vector   = np.asfarray(line_data[1:])
		im_vector   = normalize(im_vector)
		global count
		list_label[count]       = label
		list_image[:, count]    = im_vector   
		count += 1

# centroid를 정했으면 centroids 과의 거리를 계산하여 그 해당 cluster로 이동시킨다.
def image2cluster():
	global count
	global cluster
	global list_image
	global cluster_centroid
	global K
	for i in range(1,K+1):
		cluster[i] = []
	minArr = []
	for i in range(0,count):
		for j in cluster:
			minArr.append(np.linalg.norm(list_image[:, i]-cluster_centroid[j]))
		cluster_index = minArr.index(min(minArr)) + 1
		# print(minArr)
		# print(minArr.index(min(minArr)))
		cluster[cluster_index].append((list_label[i], list_image[:, i]))
		minArr = []


def centroid_select():
	global cluster_centroid
	global cluster

	for i in cluster:
		labels,images = zip(*cluster[i])
		centroid = np.sum(np.array(images), axis = 0) / len(images)
		cluster_centroid[i] = centroid 

def get_energyfunction_val():
	global cluster
	global K
	global cluster_centroid
	val = 0
	for i in cluster:
		labels,images = zip(*cluster[i])
		val =  val + np.linalg.norm(images - cluster_centroid[i])
	return val / K

def get_accuracy_val():
	global cluster
	most = 0
	labels_count = 0

	for i in cluster:
		labels,images = zip(*cluster[i])
		most = most + Counter(labels).most_common()[0][1]
		labels_count = labels_count + len(labels)
	return most/labels_count

### MAIN
K = 15
iterator = 50

cluster = {}
cluster_centroid = {}

init()
init_centroid_select_random(count,K)
image2cluster()

energy = []
accuracy = []
for x in range(0,iterator):
	centroid_select()
	image2cluster()
	energy.append(get_energyfunction_val())
	accuracy.append(get_accuracy_val())


## test data
file_data		= "mnist_test.csv"
handle_file	= open(file_data, "r")
data        		= handle_file.readlines()
handle_file.close()

num_image	= len(data)
count       	= 0 

list_image  = np.empty((size_row * size_col, num_image), dtype=float)
list_label  = np.empty(num_image, dtype=int)

for line in data:
	line_data   = line.split(',')
	label       = line_data[0]
	im_vector   = np.asfarray(line_data[1:])
	im_vector   = normalize(im_vector)

	list_label[count]       = label
	list_image[:, count]    = im_vector   
	count += 1


accuracy_test = []
image2cluster()

for x in range(0,iterator):
	centroid_select()
	image2cluster()
	accuracy_test.append(get_accuracy_val())

plt.xlabel("iterator")
plt.ylabel("accuracy")  
plt.plot(range(0,iterator), accuracy_test)
plt.show()

exit()



for i in cluster:

	im_matrix   = cluster_centroid[i].reshape((size_row, size_col))
	plt.subplot(10, 10, 1)
	plt.title(str(i)+"cluster centroid")
	plt.imshow(im_matrix, cmap='Greys', interpolation='None')
	frame   = plt.gca()
	frame.axes.get_xaxis().set_visible(False)
	frame.axes.get_yaxis().set_visible(False)

	for idx, item in enumerate(cluster[i]):
		im_matrix   = item.reshape((size_row, size_col))
		plt.subplot(10, 10, idx+11)
		plt.imshow(im_matrix, cmap='Greys', interpolation='None')

		frame   = plt.gca()
		frame.axes.get_xaxis().set_visible(False)
		frame.axes.get_yaxis().set_visible(False)
		if(idx > 50):
			break

	plt.show()


exit()








