# figure out how to add axes labels to J curve

# grabbing the following data for k-means. i will then need to normalize the data.
"""
feature 1: total amt they donated/days they've existed as a member
feature 2: avg # of days btwn donations
feature 3: # times they donated/days they've existed as a member
feature 4: avg tip/amt donated
feature 5: avg % of each pt they funded, giving them major points if they fully funded a pt or funded mowt of that ptd
"""

# either i can combine everything in pandas, or in SQL. i shld try both!

import pg, pandas as pd, numpy as np
from math import log
#conn = pg.connect(dbname="postgres", host="localhost", user="postgres")
conn = pg.connect(dbname="watsi", host="localhost", user="postgres", passwd='presario')


### Feature 1
result = conn.query("select donor_id, date_part('day',current_date-min(created_at)), sum(donation_amount)*.01 as s from contributions group by donor_id")
table1=pd.DataFrame(result.getresult())
table1=table1.set_index(table1[0])
table1[3]=table1[2].astype('float64')/table1[1]
table1=table1[3].sort_index() #series

### Feature 2
# doesn't take into account how long from today since their last donation. to do that, can use feature 3 and take total number of days since start/number of donations

table2=pd.read_pickle('./table 2 from kmeans')

"""
# subtracting one date from the subsequent one for each donor_id
result2=conn.query("select donor_id, date_trunc('day',created_at) as date from contributions order by donor_id, date")
table2a=pd.DataFrame(result2.getresult(),columns=['ID','date donated'])
table2a['date donated']=table2a['date donated'].astype('datetime64[ns]')
hier=pd.Series(table2a['date donated'].values,index=[table2a['ID'].values,table2a.index.values])

# if I cld find the diff btwn one value and the next for each donor_id using just pandas, that wld be awesome
thedict={}
for id in hier.index.levels[0]:
	print 'Running through patient',id,'out of',len(hier.index.levels[0])
	list=hier[id].tolist()
	state1=0
	for item in range(1,len(hier[id])):
		diff1=list[item]-list[item-1]
		diff=diff1.days
		state1+=diff
		state=state1/float(len(hier[id])-1)
	thedict[id]=state
table2=pd.Series(thedict)
"""

### Feature 3: # times they donated/day since their first donation
# takes into account how long since last donation
query3="select donor_id,thecount/days as freq from(select donor_id, count(*) as thecount, date_part('day',current_date-min(created_at)) as days from contributions group by donor_id) t1 order by freq desc"
result3= conn.query(query3)
table3=pd.DataFrame(result3.getresult())
table3=table3.set_index(table3[0])
table3=table3.drop([0],axis=1)
table3=table3.sort_index()[1]

### Feature 4: Tip as percentage of donation
query4="select donor_id, tip/donation as tips_as_pcg_of_donation from(select donor_id, sum(tip_amount)*.01 as tip, sum(donation_amount)*.01 as donation from contributions group by donor_id) t1"
result4= conn.query(query4)
table4=pd.DataFrame(result4.getresult())
table4=table4.set_index(table4[0])
table4=table4.drop([0],axis=1)
table4=table4.sort_index()[1]
table4=table4.astype('float64')

### Feature 5: avg % of each pt they funded, giving them major points if they fully funded a pt or funded mowt of that ptd
# put it on a log curve!

def thenorm(table):
	return (table-table.mean())/table.std()

def scale(t):
	"""Does feature scaling to [-1,1] for a table or a vertical array in a table.."""
	top=2*t-t.max()-t.min()
	bottom=t.max()-t.min()
	return top/bottom

def ci(t):
	"""t is a column from the bigtable"""
	t.name='value'
	border2=int(t.size*.9) # always 'rounds down'
	border1=int(t.size*.1)
	t=t.order()
	t=t.reset_index() # so i can take the 5th and 95th percentile values
	cut1=t[t.index>border1]
	cut2=cut1[cut1.index<border2]
	cut3=cut2.set_index(cut2.ix[:,0])
	cut4=cut3['value']
	return cut4

def thelog(t):
	t=t+1e-6
	t=log(t)
	return t
	
prenorm=pd.concat([table1,table2,table3,table4],axis=1)
#prenorm=pd.concat([ci(table1),ci(table2),ci(table3),ci(table4)],axis=1)
# this cuts off the outliers before normalization and feature scaling
prenorm+=1e-6
prenorm= prenorm.applymap(lambda x:log(x))
norm=thenorm(prenorm)
scaled=scale(norm)
values=scaled.values







# running K-means clustering on various 1D and 2D
# for some reason, doesn't work when K<=2. It works for K=>3 though.

# stuck on line 80. i know w/o classes how to get methods to run through the same data over and over, as it gets constantly updated. but i can't seem to get it to make a variable w/in a method be equal to calling another method in that class.

import numpy as np, pandas as pd

class kmeans(object):

	def __init__(self,max_iters,tries):
		"""How many times to cycle the new centroids mean and closest points in each K-means iteration?
		How many K-means iterations to try to find the one w/the lowest J, assuming random initial centroids?"""
		self.max_iters=max_iters
		self.tries=tries
	
	### Part 1: What is the data?
	# Should be in shape m,n
	def octavedata(self): #2D
		import scipy.io
		table = scipy.io.loadmat('c:/users/alex/desktop/mlclass-ex7/ex7data1.mat')
		self.x=table['X']
		self.dim=self.x.shape[1]

	def watsidata(self): #1D
		step1=np.loadtxt('c:/users/alex/desktop/python/watsi/target amt.txt')
		step2=step1.reshape(step1.size,1)
		self.x=step2*.01
		self.dim=self.x.shape[1]
		
	### Part 2: What are the initial centroids?
	def evencentroids(self): # tested for 1D, 2D, 3D
		x=self.x
		K=self.K
		dim=x.shape[1]
		therange=(x.max(axis=0)-x.min(axis=0))/float(K+1)
		c=np.arange(1,K+1)
		step1=c.reshape(K,1)*therange
		step2=step1+x.min(axis=0)
		step3=step2.reshape(K,1,dim)
		self.centroids=step3
		
	def randomcentroids(self): # tested for 1D, 2D, 3D
		x=self.x
		K=self.K
		dim=self.dim
		step1=np.random.permutation(x)
		step2=step1[:K]
		step3=step2.reshape(K,1,dim)
		self.centroids=step3
		
	### Part 3: Functions to iterate through the centroids and closest points.
	def closestcentroids(self):
		centroids=self.centroids
		X=self.x
		dim=self.dim
		if dim==1:
			dist=(centroids-X)**2 #works only for 1D data, not n*D like Octave script
			self.index=dist.argmin(axis=0)
			#print 'index is',self.index[20:50].T
		elif dim>1:	#X must have dimensions m,n. Centroids must have dimensions m,1,n. It's a 3D vector to keep it in one 'row.'
			dist=np.linalg.norm(X-centroids,axis=2).T # axis=2 for 3d through trial-and-error. same w/T! 
			#print 'step b4 index shld be\n',dist[:3]
			self.index=dist.argmin(axis=1)
			#print 'index is',self.index[20:50]
		else: print 'Can\'t understand how many dimensions this data has!'	
		
	def newcentroids(self):
		centroids=self.centroids
		idx=self.index
		X=self.x
		dim=X.shape[1]
		# if 1D doesn't work, use this: centroids[0,i,dim]=X[idx==i].mean()
		#print 'old centroids is\n',centroids
		for i in range(centroids.shape[0]):
			centroids[i][0]=X[idx==i].mean(axis=0)
		#print 'new centroids is\n',centroids
		self.centroids=centroids
	# sometimes X[idx==i] is empty when centroids are too close to each other. i just skip this calculation in run_k_means, using pass.
	
	def testoctave(self):
		self.octavedata()
		self.centroids=np.array([3,3,6,2,8,5]).reshape(3,1,2)
		self.K=3
		self.k_means_multiple(self.K)
		print self.table
	
	def testwatsi(self):
		self.watsidata()
		self.K=int(raw_input('How many centroids?\n'))
		self.randomcentroids()
		self.k_means_multiple(self.K)
		print self.table

	### Part 4: Running the K-means clustering.
	def run_k_means(self):
		"""return centroid position, how many for each, and cost"""
		centroids=self.centroids
		#print 'centroids are',centroids
		for i in range(self.max_iters):
			self.closestcentroids()
			#print 'idx for iter',i,'is',self.index
			self.newcentroids()
			#print 'centroids for iter',i,'are\n',self.centroids
		J=0
		X=self.x
		m=len(X)
		idx=self.index
		K=self.K
		dim=X.shape[1]
		for num in range(K):
			indexentries=np.nonzero(idx==num)[0] # find the index of all entries where idx==n
			values=X[indexentries] # the values in X that have the index in indesxentries
			centroid=centroids[num,0] # using one of the K centroids to do the calculation. K<=2 doesn't work here for some reason.
			J+=np.sum((values-centroid)**2)
		return [centroids.reshape((1,K,dim)),[X[idx==k].size for k in range(K)],J/m]

	### Part 5: Computing the cost.	
	def compute_cost(self,index, X):
		J=0
		m=len(X)
		X=self.x
		for num in range(K):
			indexentries=np.nonzero(index==num)[0] # find the index of all entries where idx==n
			values=X[indexentries] # the values in X that have the index in indesxentries
			centroid=centroids[0,num] # using one of the K centroids to do the calculation
			J+=np.sum((values-centroid)**2)
		return J/m

	### Part 6: Computing the centroids multiple times to find the one w/lowest cost.
	def k_means_multiple(self,K):
		"""How many centroids K to cluster the data with? The algorithm will cycle through self.max_iters times for each try, self.tries."""
		self.K=K
		list=[]
		#print 'The data is '+str(self.dim)+'-dimensional'
		#print 'There are',self.K,'centroids.'
		#print 'max_iters is',self.max_iters
		for numberoftimes in range(self.tries):
			print 'On try',numberoftimes,'out of',self.tries
			self.randomcentroids()
			try:
				atry=self.run_k_means()
			except ValueError:
				pass
			try:
				list.append(atry)
			except:
				pass
		c=['centroid position','how many for each','J']
		self.table=pd.DataFrame(list,columns=c).sort_index(by=['J']).head()

	### Part 7: Trying multiple # of centroids, or K, and plotting the cost J for each.
	def plotj(self,maxK):
		"""How many different Ks to calculate, in order to find the right number of centroids?
		Plots the cost for each number of centroids K to find the optimal number of centroids."""
		list=[]
		for k in range(2,maxK+1):
			print 'Calculating K equals',k
			#self.K=k
			self.k_means_multiple(k)
			table=self.table
			table=table.reset_index()
			table=table.drop('index',1)
			iwant=table.ix[0]
			J=iwant['J']
			#print iwant,'for K being',k
			list.append([k,J])
		toplot=pd.DataFrame(list)
		toplot=toplot.set_index(0)
		print toplot.plot(legend=False,ylim=0,xlim=2)
	
	def run(self,values):
		self.x=values
		try: self.dim=self.x.shape[1]
		except: self.dim=1
		self.K=int(raw_input('How many centroids?\n'))
		self.randomcentroids()
		print 'Randomly choosing centroids at positions\n',self.centroids
		raw_input("Press Enter to continue...")
		print 'Running K-means',self.tries,'number of tries'
		self.k_means_multiple(self.K)
		raw_input("Press Enter to continue...")
		print '\n'
		print self.table
		raw_input("Press Enter to continue...") 
		self.topgroup()
		print self.topgroup
		
	def topgroup(self):
		"""Double-check that length is same as on self.table, esp for low max_iters and tries?"""
		step1=self.table['centroid position'].reset_index().ix[0]
		step2=np.array(step1)[1] 
		self.centroids=step2.reshape(self.K,1,self.dim)		
		self.closestcentroids()
		frame=pd.DataFrame(self.index,columns=['index']) #Pandas' indexing starts from 0
		self.topgroup=frame[frame['index']==self.K-1].index+1 # -1 and +1 are bc Pandas indexes from 0
		
	### Part X: Miscellaneous	
	def alt_to_clustering(self):
		"""if i just divide the dataset by 4, do i get the same result as clustering?"""
		X=self.x
		import pandas as pd
		x=pd.DataFrame(X)
		x=pd.Series(x.ix[:,0])
		x=x.order()
		x=x.reset_index(drop=True)
		f=len(x)/4
		print 'centroids this way are',x[f],x[f*2],x[f*3]
		
	def oldtopgroup(self): # figure out how to do this in multiple dimensions
		"""This actually doesn't give me points of the last centroid, bc that wld include points before the last centroid. This just has points after the last centroid."""
		step1=self.table['centroid position'].reset_index().ix[0]
		step2=np.array(step1)[1] 
		step3=step2.reshape(step2.shape[1],step2.shape[2]) # assuming that the shape of the array that i want is a,# that i want,b; assuming that in n-D, dim is (1,k,n)
		border=step3.max() # i want all values in x above this value
		frame=pd.DataFrame(self.x) # turns the numpy array into a DataFrame so that I can keep the indexes after I filter
		topgroup1=frame[frame>border].dropna() # take all the elements of x greater than frame, show True for those which are, then drop the Falses
		topgroup2=topgroup1.index
		self.topgroup=topgroup2
		
		
# going to skip the last feature, bc there are so many 0s, the first centroid gets all of the points
# Q: are the ppl similar in the top group for each of the features? for all of the features combined?
m=values.shape[0]

a=kmeans(7,15)
a.run(values)