from sklearn.feature_selection import SelectKBest, chi2

def selectKBest(x, y, k = 250):
	transformer = SelectKBest(chi2, k = 250).fit(x, y)
	x = transformer.transform(x)
	return x