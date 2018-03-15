
# Numpy的函數簡介


```python
import numpy as np
```


```python
data = [1,2,3,4,5]
arr = np.array(data)
print(data)
arr
```

    [1, 2, 3, 4, 5]





    array([1, 2, 3, 4, 5])




```python
print(data*2)
arr*2
```

    [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]





    array([ 2,  4,  6,  8, 10])




```python
arr.dtype #arr內部的資料格式
```




    dtype('int64')



## 宣告空矩陣


```python
arr1 = np.zeros((2,4))
arr2 = np.empty((1,2))
arr3 = np.arange(5)   
print(arr1)
print(arr2)
print(arr3)
```

    [[0. 0. 0. 0.]
     [0. 0. 0. 0.]]
    [[ 0. nan]]
    [0 1 2 3 4]



```python
print( arr3.astype(np.float64) )
print( arr3 )
```

    [0. 1. 2. 3. 4.]
    [0 1 2 3 4]


## slice


```python
arr6 = np.arange(10)
arr6, arr6[5:8]
```




    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([5, 6, 7]))




```python
arr_slice = arr6[5:8]
arr_slice
```




    array([5, 6, 7])




```python
arr_slice[1] = 15
arr_slice 
```




    array([ 5, 15,  7])




```python
arr_slice[:] = 20
arr_slice 
```




    array([20, 20, 20])



## 2D


```python
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
arr2d[2], arr2d[0][2], arr2d[0, 2], arr2d[2, 0]
```




    (array([7, 8, 9]), 3, 3, 7)



# 賦值


```python
copyarr = arr2d[0].copy()
copyarr
```




    array([1, 2, 3])



# 快速地根據條件給予值 14


```python
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randint(0,2,7)
data
```




    array([0, 1, 1, 0, 0, 1, 0])




```python
data[names == 'Bob'] = 8
data
```




    array([8, 1, 1, 8, 0, 1, 0])



## Fancy indexing 16


```python
arr7 = np.array([[0., 0., 0., 0.],
		   [1., 1., 1., 1.],
		   [2., 2., 2., 2.],
		   [3., 3., 3., 3.]])
arr7[[2,0]]                         #2及0

```




    array([[2., 2., 2., 2.],
           [0., 0., 0., 0.]])




```python
arr7[2, 0]
```




    2.0




```python
arr7[[-2, -4]]   #倒過來數
```




    array([[2., 2., 2., 2.],
           [0., 0., 0., 0.]])




```python
arr8 = np.array([[0, 1, 2, 3],
		   [4, 5, 6, 7],
		   [8, 9, 10, 11],
		   [12, 13, 14, 15]])
arr8[[0], [-2]]

```




    array([2])




```python
arr8[[0, -2]]
```




    array([[ 0,  1,  2,  3],
           [ 8,  9, 10, 11]])




```python
arr8[0, -2]
```




    2



### 組合不同ndarray index的值


```python
arr8[[0], [-2]]    
```




    array([2])




```python
arr8[[0, 1, 2, 3], [-4 ,-3, -2, -1]]
```




    array([ 0,  5, 10, 15])




```python
arr8[[0, 2]]  
```




    array([[ 0,  3,  1,  2],
           [ 8, 11,  9, 10]])




```python
arr8[[0, 2]]  [:, [2, 3, 1, 2]]    #再複習
```




    array([[ 2,  3,  1,  2],
           [10, 11,  9, 10]])




```python
arr8[:, [2, 3, 1, 2]]
```




    array([[ 2,  3,  1,  2],
           [ 6,  7,  5,  6],
           [10, 11,  9, 10],
           [14, 15, 13, 14]])



## 通用函數
### 元素級數組函數
### 通常都是元素級函數的變體

### np.modf(arr)  回傳數組的小數和整數部分兩個數組 


```python
arr = np.random.randn(7)
np.modf(arr)
```




    (array([ 0.66144359, -0.62107772, -0.07982807,  0.56030198,  0.80279079,
            -0.42418625,  0.34811148]), array([ 0., -0., -1.,  0.,  0., -1.,  0.]))



## meshgrid 2d to 3d


```python
In [10]: a=np.array([1,2,3])
In [11]: b=np.array([3,4,5])
In [12]: c,d=np.meshgrid(a,b)

```


```python
d
```




    array([[3, 3, 3],
           [4, 4, 4],
           [5, 5, 5]])




```python
c
```




    array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]])



np.where(conditionArr,True,False)  #類似ifelse

## 數學與統計方法


arr.mean()==np.mean(arr)

arr.sum()

### 吃一個軸參數來指定reducing的方向


```python
arr = np.random.randn(5,4)
arr
```




    array([[ 0.83937968,  0.91126149,  1.61360889,  2.18353679],
           [ 0.94896379,  0.03180132,  1.29719309,  0.80790318],
           [-2.18746866, -1.29014261,  1.37856115,  0.06938868],
           [ 0.10753072,  1.25773386, -0.98596834, -0.46826556],
           [ 1.35291085, -0.85825169, -1.74486076, -0.30510654]])




```python
arr.mean(axis=1) #對Ｘ軸取平均
```




    array([ 1.38694671,  0.77146534, -0.50741536, -0.02224233, -0.38882703])




```python
arr.sum(axis=0) #對Ｙ軸取總和
```




    array([1.06131638, 0.05240236, 1.55853403, 2.28745655])



### arr.cumsum() #回傳結果的累進加數組


```python
arr=range(5)
arr, arr1 = np.meshgrid(arr,arr)
arr
```




    array([[0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4]])




```python
arr.cumsum() 
```




    array([ 0,  1,  3,  6, 10, 10, 11, 13, 16, 20, 20, 21, 23, 26, 30, 30, 31,
           33, 36, 40, 40, 41, 43, 46, 50])



### arr.cumprod()  #回傳結果的累乘數組


```python
arr.cumprod() 
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0])



### arr.any() 類似or
### arr.all()  類似and

## 排序

arr.sort() #從小到大排序

arr.sort(1) #從小到大排序 但按照原本陣列的形狀  1代表 每一個列都依大小排序

np.unique() #重複之值只取一個


np.in1d(arr, [2,3,6])  #arr中為[2,3,6]者為True

## 文件 I/O


```python
np.save(name,arr) # 將arr的值存到name
```


```python
np.load(name) #取出name的值
```


```python
np.savez(name,arr1,arr2)  #儲存多個物件
```


```python
np.loadtxt('arr.txt',delimiter=',') #載入txt檔
```

## 線性代數 

np.dot(arr1,arr2) # 矩陣點積

//對矩陣A，若存在一個B矩陣使得AB=BA=I，其中I為單位矩陣，則稱B為A的逆矩陣

//QR分解會將矩陣分解為一個正交矩陣(QTQ = I)和一個上三角矩陣的積



```python
from numpy.linalg import inv, qr
x=np.random.randint(0,3,size=(3,3))
print(x)
mat=x.T.dot(x)
mat
```

    [[2 0 0]
     [0 1 0]
     [2 2 1]]





    array([[8, 4, 2],
           [4, 5, 2],
           [2, 2, 1]])




```python
np.dot(x.T,x)
```




    array([[8, 4, 2],
           [4, 5, 2],
           [2, 2, 1]])




```python
print(mat)
inv(mat)  #Compute the (multiplicative) inverse of a matrix.
```

    [[8 4 2]
     [4 5 2]
     [2 2 1]]





    array([[ 0.25,  0.  , -0.5 ],
           [ 0.  ,  1.  , -2.  ],
           [-0.5 , -2.  ,  6.  ]])




```python
mat.dot(inv(mat)).round()
```




    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])




```python

```
