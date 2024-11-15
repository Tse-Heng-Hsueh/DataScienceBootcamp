#Numpy Questions
#1. Define two custom numpy arrays, say A and B. Generate two new numpy arrays by stacking A and B vertically and horizontally.
import numpy as np
A = np.array([1,2,3,4,5])
B = np.array([6,7,8,9,10])
vertical = np.vstack((A,B))
horizontal = np.hstack((A,B))
print(vertical)
print(horizontal)
#2. Find common elements between A and B. [Hint : Intersection of two sets]
common_elements = np.intersect1d(A, B)
print(common_elements)

#3. Extract all numbers from A which are within a specific range. eg between 5 and 10. [Hint: np.where() might be useful or boolean masks]
indices = np.where((A>=5) & (A<=10))
print(A[indices])
#4. Filter the rows of iris_2d that has petallength (3rd column) > 1.5 and sepallength (1st column) < 5.0
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
print(iris_2d[(iris_2d[:,2]>1.5) & (iris_2d[:,0]<5)])

#Pandas Questions
#5. From df filter the 'Manufacturer', 'Model' and 'Type' for every 20th row starting from 1st (row 0).
#```
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
#```
print(df[['Manufacturer', 'Model', 'Type']].iloc[::20])
#6. Replace missing values in Min.Price and Max.Price columns with their respective mean.
#```
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
#```
min_price_mean = df['Min.Price'].mean()
max_price_mean = df['Max.Price'].mean()
df_filled = df['Min.Price'].fillna(min_price_mean)
df_filled = df['Max.Price'].fillna(max_price_mean)
#7. How to get the rows of a dataframe with row sum > 100?
#```
df = pd.DataFrame(np.random.randint(10, 40, 60).reshape(-1, 4))
print(df[df.sum(axis = 1)>100])
