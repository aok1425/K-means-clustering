#K-means clustering using NumPy
After taking Andrew Ng's Coursera course on machine learning, I ported his K-means clustering algorithm using the Matlab code from one of the exercises. I used this algorithm to answer the question: how many groups can we split donors into?

Ideally, K-means clustering can segment the data into `n=3` number of groups, as shown here:
![intended](https://raw.githubusercontent.com/aok1425/k-means-clustering/master/images/intended.png "")

My data, however, didn't look like this. Below shows each one of the four attributes of the donors plotted against each one of the other four attributes. The histograms are for when an attribute is plotted against itself. As opposed to the previous two-dimensional plot, this data is four-dimensional.

![mine](https://raw.githubusercontent.com/aok1425/k-means-clustering/master/images/mine.png "")

##How many groups of donors are there ideally?
We can define an inaccuracy metric of having `n` groups as the distance between each one of the points belonging to a group, and the center of that group.

This graph shows the amount of inaccuracy for having `n` number of groups. 

![cost_curve](https://raw.githubusercontent.com/aok1425/k-means-clustering/master/images/cost_curve.png "")

It's a subjective judgement, but it looks like our data can be split up into `n=4` groups quite well. Once we start splitting the data into 5, 6, or more groups, the cost difference becomes trivial.

##Can the data be compressed? (using PCA)
Can we get a graph like the sample one, that shows how the data fitted neatly into 3 groups? We can't visualize 4 dimensions, but if we can compress the data into 2 or 3 dimensions, we might be able to visually distinguish the 4 groups.

Running Principal Component Analysis (this time using scikit-learn), I get this as the explained variance for each of my four features:

`[ 0.75073447  0.587651    0.0874056   0.02311994]`

That is to say, the last two attributes respectively only account for 9% and 2% of the variance in the data. The last two attributes do not affect the data as much as the first two attributes. Thus, we can reduce the four-dimensional space into two dimensions. After compression, the 2 new synthetic attributes, when plotted, look like this:

![pca](https://raw.githubusercontent.com/aok1425/k-means-clustering/master/images/pca.png "")

We can visually see that the data splits well into four groups.

##Conclusion
We compressed the four-dimensional data to visually observe that using the four attributes, donors can neatly be split into four different groups.

We can then further inspect the differences between these groups, and market differently to a specific group, for instance.