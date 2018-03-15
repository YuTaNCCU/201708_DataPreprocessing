
# coding: utf-8

# # Numpy的函數簡介

# In[2]:


import numpy as np


# In[18]:


data = [1,2,3,4,5]
arr = np.array(data)
print(data)
arr


# In[17]:


print(data*2)
arr*2


# In[21]:


arr.dtype #arr內部的資料格式


# ## 宣告空矩陣

# In[79]:


arr1 = np.zeros((2,4))
arr2 = np.empty((1,2))
arr3 = np.arange(5)   
print(arr1)
print(arr2)
print(arr3)


# In[34]:


print( arr3.astype(np.float64) )
print( arr3 )


# ## slice

# In[48]:


arr6 = np.arange(10)
arr6, arr6[5:8]


# In[49]:


arr_slice = arr6[5:8]
arr_slice


# In[50]:


arr_slice[1] = 15
arr_slice 


# In[51]:


arr_slice[:] = 20
arr_slice 


# ## 2D

# In[55]:


arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d


# In[54]:


arr2d[2], arr2d[0][2], arr2d[0, 2], arr2d[2, 0]


# # 賦值

# In[60]:


copyarr = arr2d[0].copy()
copyarr


# # 快速地根據條件給予值 14

# In[206]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randint(0,2,7)
data


# In[208]:


data[names == 'Bob'] = 8
data


# ## Fancy indexing 16

# In[66]:


arr7 = np.array([[0., 0., 0., 0.],
		   [1., 1., 1., 1.],
		   [2., 2., 2., 2.],
		   [3., 3., 3., 3.]])
arr7[[2,0]]                         #2及0


# In[62]:


arr7[2, 0]


# In[63]:


arr7[[-2, -4]]   #倒過來數


# In[74]:


arr8 = np.array([[0, 1, 2, 3],
		   [4, 5, 6, 7],
		   [8, 9, 10, 11],
		   [12, 13, 14, 15]])
arr8[[0], [-2]]


# In[75]:


arr8[[0, -2]]


# In[76]:


arr8[0, -2]


# ### 組合不同ndarray index的值

# In[78]:


arr8[[0], [-2]]    


# In[77]:


arr8[[0, 1, 2, 3], [-4 ,-3, -2, -1]]


# In[83]:


arr8[[0, 2]]  


# In[90]:


arr8[[0, 2]]  [:, [2, 3, 1, 2]]    #再複習


# In[91]:


arr8[:, [2, 3, 1, 2]]


# ## 通用函數
# ### 元素級數組函數
# ### 通常都是元素級函數的變體

# ### np.modf(arr)  回傳數組的小數和整數部分兩個數組 

# In[106]:


arr = np.random.randn(7)
np.modf(arr)


# ## meshgrid 2d to 3d

# In[131]:


a=np.array([1,2,3])
b=np.array([3,4,5])
c,d=np.meshgrid(a,b)


# In[132]:


d


# In[133]:


c


# np.where(conditionArr,True,False)  #類似ifelse

# ## 數學與統計方法
# 

# arr.mean()==np.mean(arr)

# arr.sum()

# ### 吃一個軸參數來指定reducing的方向

# In[114]:


arr = np.random.randn(5,4)
arr


# In[116]:


arr.mean(axis=1) #對Ｘ軸取平均


# In[117]:


arr.sum(axis=0) #對Ｙ軸取總和


# ### arr.cumsum() #回傳結果的累進加數組

# In[126]:


arr=range(5)
arr, arr1 = np.meshgrid(arr,arr)
arr


# In[125]:


arr.cumsum() 


# ### arr.cumprod()  #回傳結果的累乘數組

# In[127]:


arr.cumprod() 


# ### arr.any() 類似or
# ### arr.all()  類似and

# ## 排序

# arr.sort() #從小到大排序

# arr.sort(1) #從小到大排序 但按照原本陣列的形狀  1代表 每一個列都依大小排序

# np.unique() #重複之值只取一個

# 
# np.in1d(arr, [2,3,6])  #arr中為[2,3,6]者為True

# ## 文件 I/O

# In[ ]:


np.save(name,arr) # 將arr的值存到name


# In[ ]:


np.load(name) #取出name的值


# In[ ]:


np.savez(name,arr1,arr2)  #儲存多個物件


# In[ ]:


np.loadtxt('arr.txt',delimiter=',') #載入txt檔


# ## 線性代數 

# np.dot(arr1,arr2) # 矩陣點積

# //對矩陣A，若存在一個B矩陣使得AB=BA=I，其中I為單位矩陣，則稱B為A的逆矩陣
# 
# //QR分解會將矩陣分解為一個正交矩陣(QTQ = I)和一個上三角矩陣的積
# 

# In[191]:


from numpy.linalg import inv, qr
x=np.random.randint(0,3,size=(3,3))
print(x)
mat=x.T.dot(x)
mat


# In[192]:


np.dot(x.T,x)


# In[193]:


print(mat)
inv(mat)  #Compute the (multiplicative) inverse of a matrix.


# In[194]:


mat.dot(inv(mat)).round()


# In[ ]:




