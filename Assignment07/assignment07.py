import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
from pprint import pprint
from PIL import Image


def init(rows,cols,K,pix):
	label_matrix = [[0 for x in range(0,cols)] for y in range(0,rows)]
	x1 = np.zeros((rows * cols), dtype=int)
	x2 = np.zeros((rows * cols), dtype=int)
	for row in range(0,rows):
		for col in range(0,cols):
			rgb = tuple(ti/255 for ti in pix[row,col])
			label_matrix[row][col] = (random.randrange(1,K+1),rgb,[row/(rows-1),col/(cols-1)])
			#label_matrix[row][col] = (random.randrange(1,K+1),pix[row,col])
			index = (row * cols) + col
			x1[index] = row
			x2[index] = col
	return label_matrix, x1, x2


def get_locate_rgb_by_label(rows,cols,x1,x2,label_matrix):
	dic = dict()
	for i in range(0, len(x1)):
		row = x1[i]
		col = x2[i]
		data = label_matrix[row][col]
		if data[0] not in dic:
			dic[data[0]] = []
			#dic[data[0]].append([[row,col],data[1]])
			dic[data[0]].append([data[2],data[1]])
		else:
			#dic[data[0]].append([[row,col],data[1]])
			dic[data[0]].append([data[2],data[1]])

	return dic


def centroid_select(dic):
	centroid_rgb = dict()
	centroid_locate = dict()
	for i in dic:
		rgb = []
		locate = []
		for x in dic[i]:
			rgb.append(x[1])
			locate.append(x[0])
		rgb = list(map(list, rgb))
		centroid_rgb[i] = np.sum(np.array(rgb), axis = 0) / len(rgb)
		centroid_locate[i] = np.sum(np.array(locate), axis = 0)/len(locate)
	# pprint(centroid_rgb)
	# pprint(centroid_locate)

	return centroid_rgb, centroid_locate

def cluster(label_matrix,x1,x2,rgb,locate,Lambda):
	for i in range(0, len(x1)):
		row = x1[i]
		col = x2[i]
		minDist = 99999999
		label = 99999999
		for i in range(1,len(rgb) + 1):
			data = label_matrix[row][col]
			locate_dist = Lambda * (pow((locate[i][0] - data[2][0]),2) + pow((locate[i][1] - data[2][1]),2))
			rgb_dist = pow((rgb[i][0] - data[1][0]),2) + pow((rgb[i][1] - data[1][1]),2) + pow((rgb[i][2] - data[1][2]),2)
			dist = locate_dist + rgb_dist
			if minDist > dist:
				minDist = dist
				label = (i,data[1],data[2])
		label_matrix[row][col] = label


	return label_matrix

def get_energyfunction_val(dic,rgb,locate,Lambda,K): 
	val = 0
	for i in dic:
		i_locate_dist = 0
		i_rgb_dist = 0
		for x in dic[i]:
			_rgb = x[1]
			_locate = x[0]
			locate_dist = Lambda * (pow((locate[i][0] - _locate[0]),2) + pow((locate[i][1] - _locate[1]),2))
			rgb_dist = pow((rgb[i][0] - _rgb[0]),2) + pow((rgb[i][1] - _rgb[1]),2) + pow((rgb[i][2] - _rgb[2]),2)
			i_locate_dist = i_locate_dist + locate_dist
			i_rgb_dist = i_rgb_dist + rgb_dist
		dist = i_locate_dist + i_rgb_dist
		val = val + dist
	return val / K

def show_energyfunction(iter,energy):
	plt.xlabel("iterator")
	plt.ylabel("energy")
	plt.plot(range(0,iter), energy)
	plt.show()

im = Image.open('test.jpg') # Can be many different formats.
pix = im.load()
K = 5
Lambda = 8
energy = []
iterator = 10

while True:
	label_matrix , x1, x2 = init(im.size[1],im.size[0],K,pix)
	dic = get_locate_rgb_by_label(im.size[1],im.size[0],x1,x2,label_matrix)
	centroid_rgb, centroid_locate = centroid_select(dic)
	if len(list(set([tuple(set(item)) for item in [*centroid_locate.values()] ]))) == K:
		break

print("start")

for x in range(0,iterator):
	label_matrix = cluster(label_matrix,x1,x2,centroid_rgb,centroid_locate,Lambda)
	energy.append(get_energyfunction_val(dic,centroid_rgb,centroid_locate,Lambda,K))
	dic = get_locate_rgb_by_label(im.size[1],im.size[0],x1,x2,label_matrix)
	centroid_rgb, centroid_locate = centroid_select(dic)


show_energyfunction(iterator,energy)

for x in range(0,im.size[1]):
	for y in range(0,im.size[0]):
		pixel = centroid_rgb[label_matrix[x][y][0]]
		pixel = tuple(ti * 255 for ti in pixel)
		pix[x,y] = (int(pixel[0]),int(pixel[1]),int(pixel[2]))
	im.save(str(K) + "_result.png")


# cmap = plt.cm.get_cmap("hsv", K+1)
# plt.xlabel("X")
# plt.ylabel("Y")

# for i in range(0, len(x1)):
# 	row = x1[i]
# 	col = x2[i]
# 	for i in range(1, K+1):
# 		if label_matrix[row][col][0] == i:
# 			plt.scatter(row, col,c=cmap(i))

# for x in range(1,len(centroid_locate) + 1):
# 	plt.plot(centroid_locate[x][0], centroid_locate[x][1],"w*")
# plt.show()


