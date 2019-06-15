import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
from pprint import pprint
import sympy
from sympy import Symbol, solve , symbols


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

pi = 3.141592
plt.xlabel("X")
plt.ylabel("Y")
xn = np.arange(pi * -1 , pi * 1 , 0.1)
yn = [sympy.cos(v) * sympy.sin(v) for v in xn]
plt.plot(xn, yn)


coordinateX = []
coordinateY = []
for x in np.arange(pi * -1 , pi * 1 , 0.3):
	select_point_y = sympy.cos(x) * sympy.sin(x)
	select_point_y = select_point_y + random.uniform(-0.3, 0.3)
	coordinateX.append(x)
	coordinateY.append(select_point_y)
	plt.plot(x, select_point_y,'ro')

xn = np.array(coordinateX,dtype=float)
yn = np.array(coordinateY,dtype=float)


N = 35
#---------------------------
# A = np.column_stack([xn**(N-1-i) for i in range(N)])
#---------------------------
A = np.vander(xn, N)
#---------------------------
yn1 = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, yn))



#---------------------------
# p = np.asarray(yn)
# x = np.asarray(xn)
# y = np.zeros_like(xn)
# for i in range(len(p)):
#     y = y * x + p[i]
# yn = y
#---------------------------
yn1 = np.polyval(yn1, xn)
#---------------------------


# yn = np.polyfit(xn, yn, N)
# yn = np.polyval(yn, xn)
# plt.plot(xn, yn,'g-')


plt.plot(xn, yn1)

plt.show()


