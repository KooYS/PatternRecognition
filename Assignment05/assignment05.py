import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
from pprint import pprint
from collections import Counter
from PIL import Image


im = Image.open('test.jpg') # Can be many different formats.
pix = im.load()


def init(im,pix,K):
	label_matrix = [[0 for y in range(0,im.size[1])] for x in range(0,im.size[0])]
	for x in range(0,im.size[0]):
		for y in range(0,im.size[1]):
			label_matrix[x][y] = (random.randrange(1,K+1),pix[x,y])

	return label_matrix

def get_pixels_by_label(im,label_matrix):
	dic = dict()
	for x in range(0,im.size[0]):
		for y in range(0,im.size[1]):
			data = label_matrix[x][y]
			if data[0] not in dic:
				dic[data[0]] = []
				dic[data[0]].append(data[1])
			else:
				dic[data[0]].append(data[1])
	return dic

def centroid_select(pixels_by_label,cluster_centroid):
	for x in pixels_by_label:
		temp = list(map(list, pixels_by_label[x]))
		centroid = np.sum(np.array(temp), axis = 0) / len(temp)
		cluster_centroid[x] = centroid
	return cluster_centroid

def image2cluster(im,matrix,centroids):
	for x in range(0,im.size[0]):
		for y in range(0,im.size[1]):
			minDist = (99999999,99999999)
			for idx in centroids:
				dist = (np.linalg.norm(matrix[x][y][1] - centroids[idx]),idx)
				if minDist[0] > dist[0]:
					minDist = dist 
			matrix[x][y] = (minDist[1],matrix[x][y][1])
	return matrix

def get_energyfunction_val(cluster,centroids,K): 
	val = 0
	for i in cluster:
		val = val + np.linalg.norm(cluster[i] - cluster_centroid[i])
	return val / K

def show_energyfunction(iter,energy):
	plt.xlabel("iterator")
	plt.ylabel("energy")
	plt.plot(range(0,iter), energy)
	plt.show()

def save_result(im,pix,label_matrix,cluster_centroid,K):
	for x in range(0,im.size[0]):
		for y in range(0,im.size[1]):
			pixel = cluster_centroid[label_matrix[x][y][0]]
			pix[x,y] = (int(pixel[0]),int(pixel[1]),int(pixel[2]))
	im.save(str(K) + "_result.png")


iterator = 10

for K in [5,10,15,20]:
	im = Image.open('test.jpg')
    pix = im.load()
    cluster_centroid = dict()
    for i in range(1,K+1):
        cluster_centroid[i] = 0

    label_matrix = [[]]
    label_matrix = init(im,pix,K)


    energy = []
    for x in range(0,iterator):
        pixels_by_label = get_pixels_by_label(im,label_matrix)
        cluster_centroid = centroid_select(pixels_by_label,cluster_centroid)
        label_matrix = image2cluster(im,label_matrix,cluster_centroid)
        energy.append(get_energyfunction_val(pixels_by_label,cluster_centroid,K))

    print("K="+str(K)+" energy function")
    show_energyfunction(iterator,energy)
    save_result(im,pix,label_matrix,cluster_centroid,K)

