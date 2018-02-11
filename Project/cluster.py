from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import pickle
import numpy

dataframe = numpy.loadtxt("Kingston_Police_Formatted.csv",delimiter=",")
data = dataframe[:,7:9] #incident latitude & longitude
indx = dataframe[:,:6] #crime type vector

dat1 = numpy.empty(data.shape)
dat2 = numpy.empty(data.shape)
dat3 = numpy.empty(data.shape)
dat4 = numpy.empty(data.shape)
dat5 = numpy.empty(data.shape)
dat6 = numpy.empty(data.shape)

#initialize crime type counts
i1=0
i2=0
i3=0
i4=0
i5=0
i6=0

#count crime type and add crime data to dat1..dat6 array
for i in range(len(data)):
	if indx[i][0] == 1:
		dat1[i1]+=data[i][:]
		i1+=1
	elif indx[i][1] == 1:
		dat2[i2]+=data[i][:]
		i2+=1
	elif indx[i][2] == 1:
		dat3[i3]+=data[i][:]
		i3+=1
	elif indx[i][3] == 1:
		dat4[i4]+=data[i][:]
		i4+=1
	elif indx[i][4] == 1:
		dat5[i5]+=data[i][:]
		i5+=1
	else:
		dat6[i6]+=data[i][:]
		i6+=1

#for l=len(data)..0
for l in reversed(range(len(data))):
	if l > i1:
		dat1 = numpy.delete(dat1,l,0)
	if l > i2:
		dat2 = numpy.delete(dat2,l,0)
	if l > i2:
		dat3 = numpy.delete(dat3,l,0)
	if l > i4:
		dat4 = numpy.delete(dat4,l,0)
	if l > i5:
		dat5 = numpy.delete(dat5,l,0)
	if l > i6:
		dat6 = numpy.delete(dat6,l,0)

'''
Cluster crime types
'''

#determine suitable number of cluster centers using 'elbow' method
k_range = range(10,20) #try k = 10..20

#perform K-means clustering for k = 10..20
clust1 = [KMeans(n_clusters=k).fit(dat1) for k in k_range]

#get centeroids for all 10 clusters
centroids1 = [X.cluster_centers_ for X in clust1]

#distance between all datapoints and their cluster centers
k_euclid1 = [cdist(dat1,cent,'euclidean') for cent in centroids1]
dist1 = [numpy.min(ke,axis = 1) for ke in k_euclid1]

#sum squared error
wcss = [sum(d**2) for d in dist1]

#best cluster for k=10..20 is the one with minimum error 
bestclust1 = clust1[wcss.index(min(wcss))]

print("Finished cluster 1")

'''
#1 done
'''

clust2 = [KMeans(n_clusters=k).fit(dat2) for k in k_range]

centroids2 = [X.cluster_centers_ for X in clust2]

k_euclid2 = [cdist(dat2,cent,'euclidean') for cent in centroids2]
dist2 = [numpy.min(ke,axis = 1) for ke in k_euclid2]

wcss2= [sum(d**2) for d in dist2]

bestclust2 = clust2[wcss2.index(min(wcss2))]


print("Finished cluster 2")
'''
#2 done
'''
clust3 = [KMeans(n_clusters=k).fit(dat3) for k in k_range]

centroids3 = [X.cluster_centers_ for X in clust3]

k_euclid3 = [cdist(dat3,cent,'euclidean') for cent in centroids3]
dist3 = [numpy.min(ke,axis = 1) for ke in k_euclid3]

wcss3= [sum(d**2) for d in dist3]

bestclust3 = clust3[wcss3.index(min(wcss3))]


print("Finished cluster 3")
'''
#3 done
'''

clust4 = [KMeans(n_clusters=k).fit(dat4) for k in k_range]

centroids4 = [X.cluster_centers_ for X in clust4]

k_euclid4 = [cdist(dat4,cent,'euclidean') for cent in centroids4]
dist4 = [numpy.min(ke,axis = 1) for ke in k_euclid4]

wcss4 = [sum(d**2) for d in dist4]

bestclust4 = clust4[wcss4.index(min(wcss4))]


print("Finished cluster 4")
'''
#4 done
'''

clust5 = [KMeans(n_clusters=k).fit(dat5) for k in k_range]

centroids5 = [X.cluster_centers_ for X in clust5]

k_euclid5 = [cdist(dat1,cent,'euclidean') for cent in centroids5]
dist5 = [numpy.min(ke,axis = 1) for ke in k_euclid5]

wcss5 = [sum(d**2) for d in dist5]

bestclust5 = clust5[wcss5.index(min(wcss5))]


print("Finished cluster 5")
'''
#5 done
'''

clust6 = [KMeans(n_clusters=k).fit(dat6) for k in k_range]

centroids6 = [X.cluster_centers_ for X in clust6]

k_euclid6 = [cdist(dat6,cent,'euclidean') for cent in centroids6]
dist6 = [numpy.min(ke,axis = 1) for ke in k_euclid6]

wcss6 = [sum(d**2) for d in dist6]

bestclust6 = clust6[wcss6.index(min(wcss6))]


print("Finished cluster 6")
'''
#6 done
'''

pickle.dump(bestclust1,open( "clust1.p","wb"))
pickle.dump(bestclust2,open( "clust2.p","wb"))
pickle.dump(bestclust3,open( "clust3.p","wb"))
pickle.dump(bestclust4,open( "clust4.p","wb"))
pickle.dump(bestclust5,open( "clust5.p","wb"))
pickle.dump(bestclust6,open( "clust6.p","wb"))

print("All clusters saved")

