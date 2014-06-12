# going to skip the last feature, bc there are so many 0s, the first centroid gets all of the points
# Q: are the ppl similar in the top group for each of the features? for all of the features combined?\
# also, compare this to just taking the top 10% in a scalar fashion

m=values.shape[0]

kallfour1=kmeans(7,25)
kallfour1.run(values)
kallfour=kallfour1.topgroup

kone1=kmeans(7,25)
kone1.run(values[:,0].reshape(m,1))
kone=kone1.topgroup

ktwo1=kmeans(7,25)
ktwo1.run(values[:,1].reshape(m,1))
ktwo=ktwo1.topgroup

kthree1=kmeans(7,25)
kthree1.run(values[:,2].reshape(m,1))
kthree=kthree1.topgroup

scalar1=top10(scaled[0]).index
scalar2=top10(scaled[1]).index
scalar3=top10(scaled[2]).index
scalar4=top10(scaled[3]).index

# My results from Trial 1:
# Nothing in kallfour is in either kone or two. 74 people are in both kallfour and kthree. That's 9%.
# Between kallfour and scalar1, there are 57 of the same people. But 0 when I use Feature 1 using K-means!
# Kallfour and scalar2, 127.
# Kallfour and scalar3, 56.
# Kallfour and scalar4, 132.
# kone and scalar1, 18.
# ktwo and scalar2, 69.
# kthree and scalar3, 43.