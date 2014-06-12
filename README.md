#K-means clustering using NumPy
After taking Andrew Ng's Coursera course on machine learning, I ported over his K-means clustering algorithm using his Matlab code in one of the exercises. I then ran it on some data I had, data which I cannot name.

Ideally, you can easily delineate n=3 number of groups, color-coded here, if the data looks like this:
![intended](https://raw.githubusercontent.com/aok1425/k-means-clustering/master/images/intended.png "")

On my four attributes, my data looked like this however. The histograms are when an attribute is plotted against itself.
![mine](https://raw.githubusercontent.com/aok1425/k-means-clustering/master/images/mine.png "")

##How many groups could I have split my data into?
This graph shows the "cost" for having n number of groups. Roughly speaking, cost is defined as the sum of distances between the middle point for a group and each of the points belonging to that group.

![cost_curve](https://raw.githubusercontent.com/aok1425/k-means-clustering/master/images/cost_curve.png "")

It's a subjective judgement, but it looks like our data can be split up into four groups quite well.