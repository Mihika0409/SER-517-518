import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler

#Only consider level 1 and level 2.
with open('traumaNew.csv', 'rb') as inp, open('edittedTrauma.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[4].strip() == "1" or row[4].strip() == "2" or row[4].strip() == "Levels":
            writer.writerow(row)

#only consider motor vehicle accidents. 
with open('edittedTrauma.csv', 'rb') as inp, open('mvSpeedTrauma.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[13].strip() == "*NA" or row[13].strip() == "*ND" or row[13].strip() == "*BL":
            continue
        else:
            writer.writerow(row)


fields = ['Field SBP', 'Field HR', 'Field Shock Index', 'Field RR', 'Levels']

df = pd.read_csv(
    filepath_or_buffer='mvSpeedTrauma.csv',
    usecols=fields)


df.dropna(how="all", inplace=True)
print df.tail()

X = df.ix[:,1:5].values
y = df.ix[:,0].values

X_std = StandardScaler().fit_transform(X)

#find covariance matrix.
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)

cov_mat = np.cov(X_std.T)

#find eigenvectors and eigenvalues.
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

for ev in eig_vecs:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('Everything ok!')

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

#Reduce the data into 3 fields.
from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=3)
Y_sklearn = sklearn_pca.fit_transform(X_std)

print Y_sklearn

#Reduce the data into 2 fields.
sklearn_pca = sklearnPCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(X_std)

print Y_sklearn
