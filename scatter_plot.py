import pandas as pd
from pandas.tools.plotting import scatter_matrix

pd.options.display.mpl_style = 'default'


#for scatter matrix
#scatter_matrix(pd.DataFrame(X_reduced), alpha=0.2, figsize=(6, 6), diagonal='kde')

pd.DataFrame(X_reduced).plot(x=0, y=1, kind='scatter')