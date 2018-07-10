from mpi4py import MPI
import numpy as np
import time
import sys
import gc

class GlobalArray():
	def __init__(self, a,b,N):
		self.global_A = a
		self.global_B = b
		self.global_N = N 
		self.global_array = None
		self.array_row = 1
		self.array_col = N
		self.reshape = False

	def set_globArray(self, buffer_limit = None):
		if buffer_limit is None:
			buffer_limit = 10**6
			

		if self.global_N > buffer_limit:
			self.global_array= np.linspace(self.global_A, self.global_B, num=self.global_N)
			self.global_array = np.array_split(self.global_array,
				int(self.global_N/buffer_limit))
			self.array_row = int(self.global_N/buffer_limit)
			for i in range(self.array_row-1):\
				np.append(self.global_array[i],self.global_array[i+1][-1])
			self.reshape = True

		else:
			self.global_array = np.linspace(self.global_A, self.global_B, num= self.global_N)
		return self

class LocalArray():
	def __init__(self, comm):
		self.array = None
		self.split = None
		self.split_size = None
		self.split_disp = None

		self.local_array = None
		self.local_a = None
		self.local_b = None
		self.local_N = None

		self.rank = comm.Get_rank()
		self.size = comm.Get_size()
		self.name = MPI.Get_processor_name()

		self.local_dx = 0
		self.local_sum = 0

	def local_integrate(self):
		def f(x):
			func = (np.sin(np.cos(x**x) + np.sin(x**x)*(x**x)))**6
			return func
		if self.split_size[self.rank] > 0:
			self.local_dx = (self.local_b-self.local_a)/self.local_N
			self.local_sum = -self.local_dx*(f(self.local_a)+f(self.local_b))/2
			for _ in range(1,self.local_N):
				self.local_sum += f(self.local_a+self.local_dx)*self.local_dx
				self.local_a += self.local_dx
			return self.local_sum
		else:
			return 0

	def MPIScatter(self,comm):
		comm.Scatterv([self.array, self.split_size,self.split_disp, MPI.DOUBLE],
			self.local_array,
			root=0)

	def set_localArray(self,comm):
		if self.rank == 0:
			self.split = np.array_split(self.array,self.size)
			for i in range(self.size-1):
				np.append(self.split[i],self.split[i+1][-1])

			self.split_size = [len(i) for i in self.split]
			self.split_disp = np.insert(np.cumsum(self.split_size),0,0)[0:-1]

		## Broadcats the buffer
		self.split = comm.bcast(self.split,root=0)
		self.split_size = comm.bcast(self.split_size,root=0)
		self.split_disp = comm.bcast(self.split_disp,root=0)

		## Set local variable for local integration
		if self.size < len(self.array):
			self.local_a = self.split[self.rank][0]
			self.local_b = self.split[self.rank][-1]
			self.local_N = self.split_size[self.rank]
		else:
			self.local_a = self.split[self.rank]
			self.local_b = self.split[self.rank]
			self.local_N = self.split_size[self.rank]
		self.local_array = np.zeros(self.split_size[self.rank])
		return self


def main():
	## Prepare the array
	## Initilialize global vraibles
	def loop(result):

		return result

	GLOBAL_A = 0
	GLOBAL_B = np.pi/2
	GLOBAL_ITERATIONS = 10**7
	GLOBAL_ARRAY = GlobalArray(GLOBAL_A,GLOBAL_B,GLOBAL_ITERATIONS).set_globArray()
	print(GLOBAL_ARRAY.array_row)
	GLOBAL_RESULTS = 0
	if GLOBAL_ARRAY.reshape:
		for i in range(GLOBAL_ARRAY.array_row):
			## Local iterations of the sum
			comm = MPI.COMM_WORLD
			nodeArray = LocalArray(comm)
			nodeArray.array = GLOBAL_ARRAY.global_array[i]
			nodeArray.set_localArray(comm)
			GLOBAL_RESULTS += np.sum(comm.gather(nodeArray.local_integrate(),root=0))
	else:
		comm = MPI.COMM_WORLD
		nodeArray = LocalArray(comm)
		nodeArray.array = GLOBAL_ARRAY.global_array[i]
		nodeArray.set_localArray(comm)
		GLOBAL_RESULTS += np.sum(comm.gather(nodeArray.local_integrate(),root=0))

	if nodeArray.rank==0:
		print("The integral value is ", GLOBAL_RESULTS)

	## FREE MEMORY 
	del GLOBAL_ARRAY
	gc.collect()

	## Create local arrays to be distrbuted to the nodes
	## Buffer MPISIZE can handle ~10^8 data






if __name__=="__main__":
	main()