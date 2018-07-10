from mpi4py import MPI
import numpy as np 
import time
## Same function as in the sequential process
class MPIArray():
	def __init__(self):
		self.array = None
		self.split = None
		self.split_size = None
		self.split_disp = None
		self.local_array = None
		self.local_a = None
		self.local_b = None
		self.local_N = None
		self.rank = None
		self.size = None
		self.name = None
		self.local_dx = 0
		self.local_sum=0

	def set_MPI(self, comm):
		self.rank = comm.Get_rank()
		self.size = comm.Get_size()
		self.name = MPI.Get_processor_name()
		return self

	## Given local variables a, b and N
	def local_integrate(self):
		def f(x):
			func = (np.sin(np.cos(x**x) + np.sin(x**x)*(x**x)))**6
			return func
		if self.split_size[self.rank] >0:
			self.local_dx = (self.local_b-self.local_a)/self.local_N
			self.local_sum = -self.local_dx*(f(self.local_a)+f(self.local_b))/2
			for i in range(1,self.local_N):
				self.local_sum+= f(self.local_a+self.local_dx)*self.local_dx
				self.local_a+= self.local_dx
			return self.local_sum
		else:
			return 0

	## Set local properties of each array in the node
	def set_localArray(self,comm):
		if self.rank==0:
			self.split = np.array_split(self.array,self.size)
			self.split_size = [len(i) for i in self.split]
			self.split_disp = np.insert(np.cumsum(self.split_size),0,0)[0:-1]

		## Broadcast to other nodes
		self.split = comm.bcast(self.split,root=0)
		self.split_size = comm.bcast(self.split_size, root=0)
		self.disp = comm.bcast(self.split_disp,root=0)

		## Set local variable for integration 
		if self.size < len(self.array):
			self.local_a = self.split[self.rank][0]
			self.local_b = self.split[self.rank][-1]
			self.local_N = self.split_size[self.rank]
		# del self.split[:]
		else:
			self.local_a = self.split[self.rank]
			self.local_b = self.split[self.rank]
			self.local_N = self.split_size[self.rank]
		## scatter array
		self.local_array = np.zeros(self.split_size[self.rank])
		return self

	## Scatter object 
	def MPIScatter(self,comm):
		comm.Scatterv([self.array,self.split_size,self.split_disp,MPI.DOUBLE], self.local_array,root=0)

		


# Main parallelization of the program
def main(N,start_time):
	## Set properties
	comm = MPI.COMM_WORLD
	global_a = 0
	global_b = np.pi/2
	nodeArray = MPIArray()
	nodeArray.set_MPI(comm)

	nodeArray.array = np.linspace(global_a,global_b,num=N)



	## Check if there's a remainder. Add to each processor
	nodeArray.set_localArray(comm)

	## Scatter
	nodeArray.MPIScatter(comm)

	## Gather local integrate results
	newData = comm.gather(nodeArray.local_integrate(), root=0)

	# print("Rank %d got a=%f, b=%f with %d local iterations." %(rank,local_a,local_b, local_N))
	if nodeArray.rank==0:
		print("The integral value is ", np.sum(newData))
		print("Finished execution for %d iterations ... %.5fs" %(N, time.time()-start))



if __name__=="__main__":
	N = [10**5]
	# print(N)
	for n in N:
		start = time.time()
		main(n,start)


