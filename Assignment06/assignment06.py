import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np




def init(rows,cols,K):
	label_matrix = np.random.randint(K, size=(rows, cols))
	x1 = np.zeros((rows * cols), dtype=int)
	x2 = np.zeros((rows * cols), dtype=int)
	for row in range(0,rows):
		for col in range(0,cols):
			index = (row * cols) + col
			x1[index] = row
			x2[index] = col
	return label_matrix, x1, x2

def get_locate_by_label(x1,x2,label_matrix):
	dic = dict()
	for i in range(0, len(x1)):
		row = x1[i]
		col = x2[i]
		data = label_matrix[row][col]
		if data not in dic:
			dic[data] = []
			dic[data].append([row,col])
		else:
			dic[data].append([row,col])

	return dic

def centroid_select(dic):
	centroid = dict()
	for i in dic:
		centroid[i] = np.linalg.norm(dic[i], ord=1,axis=0)/len(dic[i])
		centroid[i] = [int(centroid[i][0]), int(centroid[i][1])]
	return centroid


def locate2cluster(label_matrix,x1,x2,centroid,dist):
	for i in range(0, len(x1)):
		row = x1[i]
		col = x2[i]
		minDist = 99999999
		label = 99999999
		for i in range(0,len(centroid)):
			temp = [centroid[i][0] - row, centroid[i][1] - col]
			dist = np.linalg.norm(temp,ord=dist)
			if minDist > dist:
				minDist = dist
				label = i 
		label_matrix[row][col] = label

	return label_matrix

def get_energyfunction_val(cluster,centroids,K): 
	val = 0
	for i in cluster:
		val = val + np.linalg.norm(np.subtract(cluster[i], centroid[i]))
	return val / K

def show_energyfunction(iter,energy):
	plt.xlabel("iterator")
	plt.ylabel("energy")
	plt.plot(range(0,iter), energy)
	plt.show()



# rows = 50
# cols = 45
# K = 5
# dist = 2


rows = int(input("# of Row?"))
cols = int(input("# of Columns?"))
K = int(input("# of Clusters?"))
dist = int(input("Distance type <1: L1 norm , 2: L2 norm>?"))



energy = []
iterator = 20

while True:
	label_matrix , x1, x2 = init(rows,cols,K)
	dic = get_locate_by_label(x1,x2,label_matrix)
	centroid  = centroid_select(dic)
	if len(list(set([tuple(set(item)) for item in [*centroid.values()] ]))) == K:
		break


for x in range(0,iterator):
	label_matrix = locate2cluster(label_matrix,x1,x2,centroid,dist)
	energy.append(get_energyfunction_val(dic,centroid,K))

	dic = get_locate_by_label(x1,x2,label_matrix)
	centroid  = centroid_select(dic)

cmap = plt.cm.get_cmap("hsv", K+1)
plt.xlabel("X")
plt.ylabel("Y")

for i in range(0, len(x1)):
	row = x1[i]
	col = x2[i]
	for i in range(0, K):
		if label_matrix[row][col] == i:
			plt.scatter(row, col,c=cmap(i))

for x in range(0,len(centroid)):
	plt.plot(centroid[x][0], centroid[x][1],"w*")
plt.show()

show_energyfunction(iterator,energy)

