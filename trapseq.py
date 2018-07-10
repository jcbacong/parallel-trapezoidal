## Calculate integral 

import numpy as np
import time

def f(x):
	func = np.sin(np.cos(x**x) + np.sin(x**x)*(x**x))
	return func**6

def integf(a,b,N):
	dx = (b-a)/N
	result = -dx*(f(a)+f(b))/2
	for i in range(1,N):
		result += f(a+dx)*dx
		a+= dx
	return result

def main(N):
	a = 0
	b = np.pi/2
	print("The integral value is ", integf(a,b,N))

if __name__=="__main__":
	n = [10**i for i in range(1,10)]
	for i in n:
		start = time.time()
		main(i)
		print("Execution time of 10^%d .... %.5fs" %(i, time.time()-start))
		# tot.append(time.time()-start)


	# for i in range(len(tot)):
	# 	print("Execution time of 10^%d .... %.5fs" %(i, tot[i]))
