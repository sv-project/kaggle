
* **competition** : Melbourne Housing Snapshot
* **writer** : MJ Bahmani
* **share** : @ljh0128
* **date** : 2019.06.18

# <div style="text-align: center">The Data Scientist’s Toolbox Tutorial - 2</div>

### <div style="text-align: center">CLEAR DATA. MADE MODEL.</div>
<div style="text-align:center">last update: <b>30/12/2018</b></div>


>###### Before starting to read this kernel, It is a good idea review first step: 
1. [The Data Scientist’s Toolbox Tutorial - 1](https://www.kaggle.com/mjbahmani/the-data-scientist-s-toolbox-tutorial-1)
2. <font color="red">You are in the second step</font>
3. [Mathematics and Linear Algebra](https://www.kaggle.com/mjbahmani/linear-algebra-for-data-scientists)
4. [Programming &amp; Analysis Tools](https://www.kaggle.com/mjbahmani/20-ml-algorithms-15-plot-for-beginners)
5. [Big Data](https://www.kaggle.com/mjbahmani/a-data-science-framework-for-quora)
6. [Data visualization](https://www.kaggle.com/mjbahmani/top-5-data-visualization-libraries-tutorial)
7. [Data Cleaning](https://www.kaggle.com/mjbahmani/machine-learning-workflow-for-house-prices)
8. [How to solve a Problem?](https://www.kaggle.com/mjbahmani/the-data-scientist-s-toolbox-tutorial-2)
9. [Machine Learning](https://www.kaggle.com/mjbahmani/a-comprehensive-ml-workflow-with-python)
10. [Deep Learning](https://www.kaggle.com/mjbahmani/top-5-deep-learning-frameworks-tutorial)

---------------------------------------------------------------------
You can Fork and Run this kernel on Github:
> ###### [ GitHub](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)

-------------------------------------------------------------------------------------------------------------

>  Python을 이용하여 Data Science 문제를 풀 때 사용하는 대표적인 Tools 이자 Libraries인 Numpy와 Pandas에 대해 소개하는 글이다.  
>  Numpy를 이용하여 데이터들을 계산하는 방법에 대해 다루며 Pandas를 이용하여 Dataframe이나 Series와 같은 데이터 구조와 이를 활용하는 방법에 대해 다룬다.
## 목표 ##
- 이 글을 통해 Numpy와 Pandas를 이해한다.
- 앞으로 Kaggle 문제를 풀 때 주어진 dataset을 가공하고싶은 아이디어가 떠올랐을 때 구현 방법이 생각이 안난다면 이 글을 통해 찾아가길 바란다.

 
 -----------

 <a id="top"></a> <br>
## Notebook  Content
1. [Introduction](#1)
    1. [Import](#2)
    1. [Version](#3)
1. [NumPy](#2)
    1. [Creating Arrays](#21)
    1. [How We Can Combining Arrays?](#22)
    1. [Operations](#23)
    1. [How to use Sklearn Data Set? ](#24)
    1. [Loading external data](#25)
    1. [Model Deployment](#26)
    1. [Families of ML algorithms](#27)
    1. [Prepare Features & Targets](#28)
    1. [Accuracy and precision](#29)
    1. [Estimators](#210)
    1. [Predictors](#211)
1. [Pandas](#3)
    1. [DataFrame  ](#31)
    1. [Missing values](#32)
    1. [Merging Dataframes](#33)
    1. [Making Code Pandorable](#34)
    1. [Group by](#35)
    1. [Scales](#36)
1. [Sklearn](#4)
    1. [Algorithms ](#41)
1. [conclusion](#5)
1. [References](#6)

<a id="1"></a> <br>
# 1-Introduction

이 커널은 **beginners**들이 보기에 적당하며, 물론 지식을 복습할 필요가 있는 **professionals**들에게도 권유한다.  

또한, 이 Toolbox Tutorial은 이전 버전이 있으며(  [The Data Scientist’s Toolbox Tutorial - 1](https://www.kaggle.com/mjbahmani/the-data-scientist-s-toolbox-tutorial-1) ) python을 다루는 기초적인 과정이 담겨있다.  

이번 Tutorial 에서 우리는 Kaggle 뿐만 아니라 Data science에 핵심적인 두 라이브러리에 대해 다룬다.
1. **Numpy**
1. **Pandas**

이 커널은 다음의 완벽한 tutorials 강의들을 기반으로 구성하였다:
1. [Coursera-data-science-python](https://www.coursera.org/specializations/data-science-python)
1. [Sklearn](https://scikit-learn.org)
1. [Feature Scaling with scikit-learn](http://benalexkeen.com/feature-scaling-with-scikit-learn/)
1. [https://docs.scipy.org/doc/numpy/user/quickstart.html](https://docs.scipy.org/doc/numpy/user/quickstart.html)
1. [https://pandas.pydata.org/](https://pandas.pydata.org/)
1. [https://www.tutorialspoint.com/numpy](https://www.tutorialspoint.com/numpy)
1. [python-numpy-tutorial](https://www.datacamp.com/community/tutorials/python-numpy-tutorial)

<a id="11"></a> <br>
##   1-1 Import


```python
from pandas import get_dummies
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import scipy
import numpy
import json
import sys
import csv
import os
```

<a id="12"></a> <br>
## 1-2 Version


```python
print('scipy: {}'.format(scipy.__version__))
print('seaborn: {}'.format(sns.__version__))
print('pandas: {}'.format(pd.__version__))
print('numpy: {}'.format(np.__version__))
print('Python: {}'.format(sys.version))
```

    scipy: 1.1.0
    seaborn: 0.9.0
    pandas: 0.23.4
    numpy: 1.16.3
    Python: 3.6.6 |Anaconda, Inc.| (default, Oct  9 2018, 12:34:16) 
    [GCC 7.3.0]
    

<a id="13"></a> <br>
## 1-3 Setup

A few tiny adjustments for better **code readability**


```python
warnings.filterwarnings('ignore')
%precision 2
```




    '%.2f'



<a id="14"></a> <br>
## 1-4 Import DataSets


```python
    hp_train=pd.read_csv('../input/melb_data.csv')
```

<a id="2"></a> <br>
# 2- NumPy
Numpy는 오픈 소스 라이브러리로 이를 통해 정말 많은 산술 연산을 수행할 수 있다. 자세한 내용은, 홈페이지에 있으며 [page](http://www.numpy.org/) 여기서는 핵심적인 기능들에 대해 소개한다.

<img src='https://scipy-lectures.org/_images/numpy_indexing.png' width=400 heght=400>
[**Image Credit**](https://scipy-lectures.org/intro/numpy/array_object.html)

**Numpy의 가장 대표적인 특징**:
1. **Numerical Python을 의미** : stands for Numerical Python
1. **산술 연산과 논리 연산을 사용가능** : Use for mathematical and logical operations
1. **선형 대수와 관련된 연산들을 사용가능** : Operations related to linear algebra
1. **인덱싱과 슬라이싱에 매우 효율적** : Numpy is good  for  indexing and slicing

For a fast start you can check this [cheatsheat](https://www.datacamp.com/community/blog/python-numpy-cheat-sheet) too.


```python
import numpy as np
```

<a id="21"></a> <br>
## 2-1 배열(Array)를 생성하는 방법

`np.array` 로 배열을 생성한다


```python
mylist = [1, 2, 3]
myarray = np.array(mylist)
myarray.shape
```




    (3,)



<img src='http://community.datacamp.com.s3.amazonaws.com/community/production/ckeditor_assets/pictures/332/content_arrays-axes.png' width=500 heght=500>
[**Image Credit**](https://www.datacamp.com/community/tutorials/python-numpy-tutorial)


```python
myarray.shape
```




    (3,)




`resize` 는 배열의 모양과 크기를 바꾼다. 이 때 해당 배열 변수에 그대로 적용이 된다**(in-place)**.


```python
myarray.resize(3, 3)
myarray
```




    array([[1, 2, 3],
           [0, 0, 0],
           [0, 0, 0]])




`ones` 는 **1**로 채워진 배열을 생성하고, 인자로 배열의 모양과 자료형을 입력해준다.


```python
np.ones((3, 2))
```




    array([[1., 1.],
           [1., 1.],
           [1., 1.]])




`zeros` 는 **0**으로 채워진 배열을 생성하고, 인자로 배열의 모양과 자료형을 입력해준다.


```python
np.zeros((2, 3))
```




    array([[0., 0., 0.],
           [0., 0., 0.]])




`eye` 는 대각 성분의 값이 1이고 나머지 값들은 0인 **2차원**의 단위 행렬(identity matrix)을 생성한다. 인자로 n을 입력하게 되면 nxn 의 단위 행렬이 만들어진다.


```python
np.eye(3)
```




    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])




`diag` 는 입력 받은 2차원 행렬의 대각 성분 값을 추출하거나, 입력 받은 1차원 행렬로부터 대각 행렬을 생성한다.


```python
np.diag(myarray)
```




    array([1, 0, 0])




```python
np.diag(np.diag(myarray))
```




    array([[1, 0, 0],
           [0, 0, 0],
           [0, 0, 0]])




list 반복 연산을 이용해서 `np.array` 로 배열을 생성할 수도 있다. 또는 `np.tile` 로 같은 작동을 할 수 있다.


```python
np.array([1, 2, 3] * 3)
```




    array([1, 2, 3, 1, 2, 3, 1, 2, 3])




```python
np.tile([1,2,3],3)
```




    array([1, 2, 3, 1, 2, 3, 1, 2, 3])




`repeat` 을 이용해서 list가 반복되는 배열을 만들 수도 있다. 이 때, `np.tile`와 다른 점은 각 원소를 바로 옆에 입력 받은 개수 만큼 반복시켜서 배열을 생성한다.

- 인자에 'axis=' 도 추가가 가능하며 이에 대한 추가 자료 필요쓰


```python
np.repeat([1, 2, 3], 3)
```




    array([1, 1, 1, 2, 2, 2, 3, 3, 3])



<a id="22"></a> <br>
## 2-2 배열들을 합치는 방법 (**vstack, hstack**)
[docs.scipy.org](https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html)
###### [Go to top](#top)


```python
p = np.ones([2, 3], int)
p
```




    array([[1, 1, 1],
           [1, 1, 1]])




`vstack` 은 수직(vertically) 방향으로 배열을 합친다 (row wise) .


```python
np.vstack([p, 2*p])
```




    array([[1, 1, 1],
           [1, 1, 1],
           [2, 2, 2],
           [2, 2, 2]])




`hstack` 은 수평(horizontally) 방향으로 배열을 합친다 (column wise) .


```python
np.hstack([p, 2*p])
```




    array([[1, 1, 1, 2, 2, 2],
           [1, 1, 1, 2, 2, 2]])



<a id="23"></a> <br>
## 2-3 기본 연산 (Operations)
for learning numpy operator, this good idea to get a visit in this [page](http://scipy-lectures.org/intro/numpy/operations.html)
<br><br>
아래 그림에서 왼쪽의 세 방식의 입력은 같은 결과를 가져온다는 것을 그림으로 나타낸 것이다.
<img src='http://scipy-lectures.org/_images/numpy_broadcasting.png'>
[Image Credit](http://scipy-lectures.org/intro/numpy/operations.html)
###### [Go to top](#top)

`+`, `-`, `*`, `/` and `**` 을 이용해 원소 별로 (**element wise**) 덧셈, 뺄셈, 곱셈, 급수(power) 연산을 할 수 있다.


```python
x=np.array([1, 2, 3])
y=np.array([4, 5, 6])
```


```python
print(x + y) # elementwise addition     [1 2 3] + [4 5 6] = [5  7  9]
print(x - y) # elementwise subtraction  [1 2 3] - [4 5 6] = [-3 -3 -3]
```

    [5 7 9]
    [-3 -3 -3]
    


```python
print(x * y) # elementwise multiplication  [1 2 3] * [4 5 6] = [4  10  18]
print(x / y) # elementwise divison         [1 2 3] / [4 5 6] = [0.25  0.4  0.5]
```

    [ 4 10 18]
    [0.25 0.4  0.5 ]
    


```python
print(x**2) # elementwise power  [1 2 3] ^2 =  [1 4 9]
```

    [1 4 9]
    


**내적(Dot Product):**  

$ \begin{bmatrix}x_1 \ x_2 \ x_3\end{bmatrix}
\cdot
\begin{bmatrix}y_1 \\ y_2 \\ y_3\end{bmatrix}
= x_1 y_1 + x_2 y_2 + x_3 y_3$


```python
x.dot(y) # dot product  1*4 + 2*5 + 3*6
```




    32




```python
z = np.array([y, y**2])
print(len(z)) # number of rows of array
print(z) # array z
```

    2
    [[ 4  5  6]
     [16 25 36]]
    


**Transposing** (전치)는 배열의 모양 (dimensions)을 변경시킨다. 


```python
z = np.array([y, y**2])
z
```




    array([[ 4,  5,  6],
           [16, 25, 36]])




`z`은 transpoing 이전에 `(2,3)` 의 모양을 갖는다.


```python
z.shape
```




    (2, 3)




`.T` 를 사용해 transpose를 할 수 있다.


```python
z.T
```




    array([[ 4, 16],
           [ 5, 25],
           [ 6, 36]])



<br>
수행하게 되면 row와 column이 서로 바뀌게 된다.


```python
z.T.shape
```




    (3, 2)




`.dtype` 는 배열의 원소들의 자료형을 볼 수 있다. 


```python
z.dtype
```




    dtype('int64')




`.astype` 은 배열의 원소들의 자료형을 특정 값으로 변환시킬 수 있다.


```python
z = z.astype('f')
z.dtype
```




    dtype('float32')



<a id="24"></a> <br>
## 2-4 수학 연산 (Math Functions)
For learning numpy math function, this good idea to get a visit in this [page](https://www.geeksforgeeks.org/numpy-mathematical-function/)
<img src='http://s8.picofile.com/file/8353147492/numpy_math.png'>
[Image Credit](https://www.geeksforgeeks.org/numpy-mathematical-function/)

###### [Go to top](#top)

Numpy 는 배열에 적용시킬 수 있는 정말 다양한 built in math functions을 갖고 있다.


```python
myarray = np.array([-4, -2, 1, 3, 5])
```


```python
myarray.sum()
```




    3




```python
myarray.max()
```




    5




```python
myarray.min()
```




    -4




```python
myarray.mean()
```




    0.6




```python
myarray.std()
```




    3.2619012860600183




`argmax` and `argmin` 은 배열에서 최댓값과 최솟값을 갖는 원소의 위치(index)를 반환한다. 


```python
myarray.argmax()
```




    4




```python
myarray.argmin()
```




    0



<a id="25"></a> <br>

## 2-5 인덱싱 / 슬라이싱 (Indexing / Slicing)
For learning numpy Indexing / Slicing , this good idea to get a visit in this [page](https://www.stechies.com/numpy-indexing-slicing/)
<img src='http://s8.picofile.com/file/8353147750/numpy_math2.png'>
[Image Credit](https://www.stechies.com/numpy-indexing-slicing/)
###### [Go to top](#top)


```python
myarray = np.arange(13)**2
myarray
```




    array([  0,   1,   4,   9,  16,  25,  36,  49,  64,  81, 100, 121, 144])




`[ ]` (bracket) 을 이용해 특정 위치의 값을 볼 수 있다.


```python
myarray[0], myarray[4], myarray[-1]
```




    (0, 16, 144)




`:` 는 범위를 지정할 때 사용한다. `array[start:stop]`


`start` 와 `stop` 은 비워둘 수 있으며 그 때는 배열의 시작과 끝을 의미한다.


```python
myarray[1:5]
```




    array([ 1,  4,  9, 16])




```python
myarray[:5]
```




    array([ 0,  1,  4,  9, 16])




```python
myarray[1:]
```




    array([  1,   4,   9,  16,  25,  36,  49,  64,  81, 100, 121, 144])



<br>
음수 값은 배열의 뒤에서 부터 몇 번 째인지를 의미한다.


```python
myarray[-4:]
```




    array([ 81, 100, 121, 144])




두 번 째 `:` 는 step-size를 의미한다. `array[start:stop:stepsize]` 사용하지 않을 경우 1로 적용한다.

배열의 뒤에서 5번 째부터 시작하여 배열의 끝까지 뒤로 두 칸씩 돌아가면서 존재하는 원소들을 알아보고 싶은 경우는 다음과 같이 코딩할 수 있다.


```python
myarray[-5::-2]
```




    array([64, 36, 16,  4,  0])



<br>
**multidimensional array.**


```python
r = np.arange(36)
r.resize((6, 6))
r
```




    array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17],
           [18, 19, 20, 21, 22, 23],
           [24, 25, 26, 27, 28, 29],
           [30, 31, 32, 33, 34, 35]])



<br>
다차원 배열에서 다음과 같이 slicing을 할 수 있다: `array[row, column]`.


```python
r[2, 2]
```




    14




그리고 `:` 을 이용하여 row 또는 column의 범위를 지정하여 slicing을 할 수 있다.


```python
r[3, 3:6]
```




    array([21, 22, 23])



<br>
배열의 2행 이전까지, 그리고 마지막 열 이전까지 존재하는 원소들을 slicing 하고싶은 경우


```python
r[:2, :-1]
```




    array([[ 0,  1,  2,  3,  4],
           [ 6,  7,  8,  9, 10]])



<br>
마지막 행의 짝수 열의 값들을 slicing 하고싶은 경우 (각 행과 열은 0부터 시작한다고 생각하자)


```python
r[-1, ::2]
```




    array([30, 32, 34])



<br>
또 우리는 조건식을 이용하여 indexing을 할 수 있다.
<br><br>
배열에 조건식을 적용시키면 해당 조건을 만족하는 원소의 위치에 True, 만족하지 않으면 False 값을 갖는 배열이 만들어지며 이를 이용해 indexing이 가능하다.
<br><br>
예를 들어, 30보다 큰 값을 갖는 원소들을 indexing 하려는 경우 (Also see `np.where`)


```python
r>30
```




    array([[False, False, False, False, False, False],
           [False, False, False, False, False, False],
           [False, False, False, False, False, False],
           [False, False, False, False, False, False],
           [False, False, False, False, False, False],
           [False,  True,  True,  True,  True,  True]])




```python
r[r > 30]
```




    array([31, 32, 33, 34, 35])



<br>
30보다 큰 값을 갖는 원소들을 찾아 그 값을 30으로 바꾸고 싶은 경우
###### [Go to top](#top)


```python
r[r > 30] = 30
r
```




    array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17],
           [18, 19, 20, 21, 22, 23],
           [24, 25, 26, 27, 28, 29],
           [30, 30, 30, 30, 30, 30]])



<a id="26"></a> <br>
## 2-6 복사 (Copying Data)

Be careful with copying and modifying arrays in NumPy!
<br><br>
자 다음 글을 숙지하고 배열에서 우리가 값을 복사하고 수정할 때 조심하자!!


`r2` is a slice of `r`


```python
r2 = r[:3,:3]
r2
```




    array([[ 0,  1,  2],
           [ 6,  7,  8],
           [12, 13, 14]])



<br>
Set this slice's values to zero ([:] selects the entire array)



```python
r2[:] = 0
r2
```




    array([[0, 0, 0],
           [0, 0, 0],
           [0, 0, 0]])




`r` has also been changed!

> 우리가 단순히 값을 slicing 해와서 복사한 배열에서 값을 수정하게될 경우, 원본 배열의 값도 같이 수정이 되어버린다.<br>
> 따라서 필요한 경우에 따라 다르겠지만 그것을 원치 않는다면 다른 방법으로 Data를 복사하즈아.


```python
r
```




    array([[ 0,  0,  0,  3,  4,  5],
           [ 0,  0,  0,  9, 10, 11],
           [ 0,  0,  0, 15, 16, 17],
           [18, 19, 20, 21, 22, 23],
           [24, 25, 26, 27, 28, 29],
           [30, 30, 30, 30, 30, 30]])




`r.copy` 를 이용하게 되면 원본 배열에 영향을 끼치지 않는 값만 복사된 새 배열을 만들 수 있다.


```python
r_copy = r.copy()
r_copy
```




    array([[ 0,  0,  0,  3,  4,  5],
           [ 0,  0,  0,  9, 10, 11],
           [ 0,  0,  0, 15, 16, 17],
           [18, 19, 20, 21, 22, 23],
           [24, 25, 26, 27, 28, 29],
           [30, 30, 30, 30, 30, 30]])



<br>
Now when r_copy is modified, r will not be changed.


```python
r_copy[:] = 10
print(r_copy, '\n')
print(r)
```

    [[10 10 10 10 10 10]
     [10 10 10 10 10 10]
     [10 10 10 10 10 10]
     [10 10 10 10 10 10]
     [10 10 10 10 10 10]
     [10 10 10 10 10 10]] 
    
    [[ 0  0  0  3  4  5]
     [ 0  0  0  9 10 11]
     [ 0  0  0 15 16 17]
     [18 19 20 21 22 23]
     [24 25 26 27 28 29]
     [30 30 30 30 30 30]]
    

<a id="27"></a> <br>
## 2-7 Iterating Over Arrays

Let's create a new 4 by 3 array of random numbers 0-9.


```python
test = np.random.randint(0, 10, (4,3))
test
```




    array([[1, 1, 1],
           [4, 9, 7],
           [9, 5, 6],
           [4, 9, 5]])



<br>
Iterate by row:


```python
for row in test:
    print(row)
```

    [1 1 1]
    [4 9 7]
    [9 5 6]
    [4 9 5]
    

<br>
Iterate by index:


```python
for i in range(len(test)):
    print(test[i])
```

    [1 1 1]
    [4 9 7]
    [9 5 6]
    [4 9 5]
    

<br>
Iterate by row and index:


```python
for i, row in enumerate(test):
    print('row', i, 'is', row)
```

    row 0 is [1 1 1]
    row 1 is [4 9 7]
    row 2 is [9 5 6]
    row 3 is [4 9 5]
    


`zip` 을 이용하면 다수의 iterable 한 객체들의 개별 원소들을 동시에 반복적으로 셀 (iterable) 수 있다.


```python
test2 = test**2
test2
```




    array([[ 1,  1,  1],
           [16, 81, 49],
           [81, 25, 36],
           [16, 81, 25]])




```python
for i, j in zip(test, test2):
    print(i,'+',j,'=',i+j)
```

    [1 1 1] + [1 1 1] = [2 2 2]
    [4 9 7] + [16 81 49] = [20 90 56]
    [9 5 6] + [81 25 36] = [90 30 42]
    [4 9 5] + [16 81 25] = [20 90 30]
    

<a id="28"></a> <br>
## 2-8 The Series Data Structure
One-dimensional ndarray with axis labels (including time series)
<br><br>
Series Data란 1차원 배열 데이터로 0 ~ n-1 이 아닌 축 (axis label) 이 존재하고 그 값으로 indexing 할 수 있는 데이터 구조이다.
<br><br>
For learning Series Data Structure , this good idea to get a visit in this [page](https://www.kdnuggets.com/2017/01/pandas-cheat-sheet.html)
<br><br>
**Series 와 DataFrame**
<img src='https://www.kdnuggets.com/wp-content/uploads/pandas-02.png'>
[Image Credit](https://www.kdnuggets.com/2017/01/pandas-cheat-sheet.html)


```python
animals = ['Tiger', 'Bear', 'Moose']
pd.Series(animals)
```


```python
numbers = [1, 2, 3]
pd.Series(numbers)
```


```python
animals = ['Tiger', 'Bear', None]
pd.Series(animals)
```


```python
numbers = [1, 2, None]
pd.Series(numbers)
```


```python
import numpy as np
np.nan == None
```


```python
np.nan == np.nan
```


```python
np.isnan(np.nan)
```


```python
sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports)
s
```


```python
s.index
```


```python
s = pd.Series(['Tiger', 'Bear', 'Moose'], index=['India', 'America', 'Canada'])
s
```


```python
sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports, index=['Golf', 'Sumo', 'Hockey'])
s
```

<a id="29"></a> <br>
# 2-9 Querying a Series


```python
sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports)
s
```


```python
s.iloc[3]
```


```python
s.loc['Golf']
```


```python
s[3]
```


```python
s['Golf']
```


```python
sports = {99: 'Bhutan',
          100: 'Scotland',
          101: 'Japan',
          102: 'South Korea'}
s = pd.Series(sports)
```


```python
s = pd.Series([100.00, 120.00, 101.00, 3.00])
s
```


```python
total = 0
for item in s:
    total+=item
print(total)
```


```python
total = np.sum(s)
print(total)
```


```python
#this creates a big series of random numbers
s = pd.Series(np.random.randint(0,1000,10000))
s.head()
```


```python
len(s)
```


```python
%%timeit -n 100
summary = 0
for item in s:
    summary+=item
```


```python
%%timeit -n 100
summary = np.sum(s)
```


```python
s+=2 #adds two to each item in s using broadcasting
s.head()
```


```python
for label, value in s.iteritems():
    s.set_value(label, value+2)
s.head()
```


```python
%%timeit -n 10
s = pd.Series(np.random.randint(0,1000,100))
for label, value in s.iteritems():
    s.loc[label]= value+2
```


```python
%%timeit -n 10
s = pd.Series(np.random.randint(0,1000,100))
s+=2

```


```python
s = pd.Series([1, 2, 3])
s.loc['Animal'] = 'Bears'
s
```

<a id="210"></a> <br>
## 2-10 Distributions in Numpy
###### [Go to top](#top)


```python
np.random.binomial(1, 0.5)
```


```python
np.random.binomial(1000, 0.5)/1000
```


```python
chance_of_tornado = 0.01/100
np.random.binomial(100000, chance_of_tornado)
```


```python
chance_of_tornado = 0.01

tornado_events = np.random.binomial(1, chance_of_tornado, 1000000)
    
two_days_in_a_row = 0
for j in range(1,len(tornado_events)-1):
    if tornado_events[j]==1 and tornado_events[j-1]==1:
        two_days_in_a_row+=1

print('{} tornadoes back to back in {} years'.format(two_days_in_a_row, 1000000/365))
```


```python
np.random.uniform(0, 1)
```


```python
np.random.normal(0.75)
```


```python
distribution = np.random.normal(0.75,size=1000)

np.sqrt(np.sum((np.mean(distribution)-distribution)**2)/len(distribution))
```


```python
np.std(distribution)
```

<a id="3"></a> <br>
## 3- Pandas
Pandas is capable of many tasks including  [Based on this [page](https://medium.com/dunder-data/how-to-learn-pandas-108905ab4955)]:

1. Reading/writing many different data formats
1. Selecting subsets of data
1. Calculating across rows and down columns
1. Finding and filling missing data
1. Applying operations to independent groups within the data
1. Reshaping data into different forms
1. Combing multiple datasets together
1. Advanced time-series functionality
1. Visualization through matplotlib and seaborn

###### [Go to top](#top)


```python

purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})
df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])
df.head()
```


```python
df.loc['Store 2']
```


```python
type(df.loc['Store 2'])
```


```python
df.loc['Store 1']
```


```python
df.loc['Store 1', 'Cost']
```


```python
df.T
```


```python
df.T.loc['Cost']
```


```python
df['Cost']
```


```python
df.loc['Store 1']['Cost']
```


```python
df.loc[:,['Name', 'Cost']]
```


```python
df.drop('Store 1')
```


```python
df
```


```python
copy_df = df.copy()
copy_df = copy_df.drop('Store 1')
copy_df
```


```python
copy_df.drop
```


```python
del copy_df['Name']
copy_df
```


```python
df['Location'] = None
df
```


```python
costs = df['Cost']
costs
```


```python
costs+=2
costs
```


```python
df
```

<a id="31"></a> <br>
# 3-1 Dataframe

As a Data Scientist, you'll often find that the data you need is not in a single file. It may be spread across a number of text files, spreadsheets, or databases. You want to be able to import the data of interest as a collection of DataFrames and figure out how to combine them to answer your central questions.
###### [Go to top](#top)


```python
df = pd.read_csv('../input/melb_data.csv')
df.head()
```


```python
df.columns
```


```python
# Querying a DataFrame
```


```python
df['Price'] > 10000000
```


```python
only_SalePrice = df.where(df['Price'] > 0)
only_SalePrice.head()
```


```python
only_SalePrice['Price'].count()
```


```python
df['Price'].count()
```


```python
only_SalePrice = only_SalePrice.dropna()
only_SalePrice.head()
```


```python
only_SalePrice = df[df['Price'] > 0]
only_SalePrice.head()
```


```python
len(df[(df['Price'] > 0) | (df['Price'] > 0)])
```


```python
df[(df['Price'] > 0) & (df['Price'] == 0)]
```

<a id="311"></a> <br>
## 3-1-1 Dataframes


```python
df.head()
```


```python
df['SalePrice'] = df.index
df = df.set_index('SalePrice')
df.head()
```


```python

df = df.reset_index()
df.head()
```

<a id="32"></a> <br>
# 3-2 Missing values



```python
df = pd.read_csv('../input/melb_data.csv')
```


```python
df.fillna
```


```python
df = df.fillna(method='ffill')
df.head()
```

<a id="33"></a> <br>
# 3-3 Merging Dataframes
For learning Merging Dataframes , this is a good idea to give a visit in this [page](https://www.ryanbaumann.com/blog/2016/4/30/python-pandas-tosql-only-insert-new-rows)
<img src='https://static1.squarespace.com/static/54bb1957e4b04c160a32f928/t/5724fd0bf699bb5ad6432150/1462041871236/?format=750w'>
[Image Credit](https://www.ryanbaumann.com/blog/2016/4/30/python-pandas-tosql-only-insert-new-rows)



```python
df = pd.DataFrame([{'Name': 'MJ', 'Item Purchased': 'Sponge', 'Cost': 22.50},
                   {'Name': 'Kevyn', 'Item Purchased': 'Kitty Litter', 'Cost': 2.50},
                   {'Name': 'Filip', 'Item Purchased': 'Spoon', 'Cost': 5.00}],
                  index=['Store 1', 'Store 1', 'Store 2'])
df
```


```python
df['Date'] = ['December 1', 'January 1', 'mid-May']
df
```


```python
df['Delivered'] = True
df
```


```python
df['Feedback'] = ['Positive', None, 'Negative']
df
```


```python
adf = df.reset_index()
adf['Date'] = pd.Series({0: 'December 1', 2: 'mid-May'})
adf
```

<a id="34"></a> <br>
# 3-4 Making Code Pandorable
based on this amazing **[Article](https://www.datacamp.com/community/tutorials/pandas-idiomatic)**
1. Indexing with the help of **loc** and **iloc**, and a short introduction to querying your DataFrame with query();
1. Method Chaining, with the help of the pipe() function as an alternative to nested functions;
1. Memory Optimization, which you can achieve through setting data types;
1. groupby operation, in the naive and Pandas way; and
1. Visualization of your DataFrames with Matplotlib and Seaborn.


```python
df = pd.read_csv('../input/melb_data.csv')
```


```python
df.head()
```

<a id="35"></a> <br>
## 3-5 Group by


```python
df = df[df['Price']>500000]
df
```

<a id="36"></a> <br>
## 3-6 Scales



```python
df = pd.DataFrame(['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D'],
                  index=['excellent', 'excellent', 'excellent', 'good', 'good', 'good', 'ok', 'ok', 'ok', 'poor', 'poor'])
df.rename(columns={0: 'Grades'}, inplace=True)
df
```


```python
df['Grades'].astype('category').head()
```


```python
grades = df['Grades'].astype('category',
                             categories=['D', 'D+', 'C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+'],
                             ordered=True)
grades.head()
```


```python
grades > 'C'
```

<a id="361"></a> <br>
## 3-6-1 Select

To select rows whose column value equals a scalar, some_value, use ==:


```python
df.loc[df['Grades'] == 'A+']

```

To select rows whose column value is in an iterable, some_values, use **isin**:


```python
df_test = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 4, 7]})
df_test.isin({'A': [1, 3], 'B': [4, 7, 12]})
```

Combine multiple conditions with &:


```python
df.loc[(df['Grades'] == 'A+') & (df['Grades'] == 'D')]

```

To select rows whose column value does not equal some_value, use !=:



```python

df.loc[df['Grades'] != 'B+']

```

isin returns a boolean Series, so to select rows whose value is not in some_values, negate the boolean Series using ~:



```python
df_test = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 4, 7]})
```


```python
df_test.loc[~df_test['A'].isin({'A': [1, 3], 'B': [4, 7, 12]})]
```

<a id="37"></a> <br>
## 3-7 Date Functionality
###### [Go to top](#top)

<a id="371"></a> <br>
### 3-7-1 Timestamp


```python
pd.Timestamp('9/1/2016 10:05AM')
```

<a id="372"></a> <br>
### 3-7-2 Period


```python
pd.Period('1/2016')
```


```python
pd.Period('3/5/2016')
```

<a id="373"></a> <br>
### 3-7-3 DatetimeIndex


```python
t1 = pd.Series(list('abc'), [pd.Timestamp('2016-09-01'), pd.Timestamp('2016-09-02'), pd.Timestamp('2016-09-03')])
t1
```


```python
type(t1.index)
```

<a id="374"></a> <br>
### 3-7-4 PeriodIndex


```python
t2 = pd.Series(list('def'), [pd.Period('2016-09'), pd.Period('2016-10'), pd.Period('2016-11')])
t2
```


```python
type(t2.index)
```

<a id="38"></a> <br>
## 3-8 Converting to Datetime


```python
d1 = ['2 June 2013', 'Aug 29, 2014', '2015-06-26', '7/12/16']
ts3 = pd.DataFrame(np.random.randint(10, 100, (4,2)), index=d1, columns=list('ab'))
ts3
```


```python
ts3.index = pd.to_datetime(ts3.index)
ts3
```


```python
pd.to_datetime('4.7.12', dayfirst=True)
```


```python
pd.Timestamp('9/3/2016')-pd.Timestamp('9/1/2016')
```

<a id="381"></a> <br>
### 3-8-1 Timedeltas


```python
pd.Timestamp('9/3/2016')-pd.Timestamp('9/1/2016')
```


```python
pd.Timestamp('9/2/2016 8:10AM') + pd.Timedelta('12D 3H')
```

<a id="382"></a> <br>
### 3-8-2 Working with Dates in a Dataframe



```python
dates = pd.date_range('10-01-2016', periods=9, freq='2W-SUN')
dates
```


```python
df.index.ravel
```




<a id="4"></a> <br>
# 4- Sklearn 
[sklearn has following feature](https://scikit-learn.org/stable/):
1. Simple and efficient tools for data mining and data analysis
1. Accessible to everybody, and reusable in various contexts
1. Built on NumPy, SciPy, and matplotlib
1. Open source, commercially usable - BSD license

<a id="41"></a> <br>
## 4-1 Algorithms

**Supervised learning**:

1. Linear models (Ridge, Lasso, Elastic Net, ...)
1. Support Vector Machines
1. Tree-based methods (Random Forests, Bagging, GBRT, ...)
1. Nearest neighbors 
1. Neural networks (basics)
1. Gaussian Processes
1. Feature selection

**Unsupervised learning**:

1. Clustering (KMeans, Ward, ...)
1. Matrix decomposition (PCA, ICA, ...)
1. Density estimation
1. Outlier detection

__Model selection and evaluation:__

1. Cross-validation
1. Grid-search
1. Lots of metrics

_... and many more!_ (See our [Reference](http://scikit-learn.org/dev/modules/classes.html))

For learning this section please give a visit on [this kernel](https://www.kaggle.com/mjbahmani/20-ml-algorithms-15-plot-for-beginners)

<a id="7"></a> <br>
# 7- conclusion
After the first version of this kernel, in the second edition, we introduced Numpy & Pandas. in addition, we examined each one in detail. This kernel is finished due to the large size and you can follow the discussion in the my other kernel.

>###### you may  be interested have a look at it: [**10-steps-to-become-a-data-scientist**](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)


---------------------------------------------------------------------
you can Fork and Run this kernel on Github:
> ###### [ GitHub](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)

-------------------------------------------------------------------------------------------------------------

 **I hope you find this kernel helpful and some <font color="red"><b>UPVOTES</b></font> would be very much appreciated**
 
 -----------

<a id="8"></a> <br>
# 8- References & Credits
1. [Coursera](https://www.coursera.org/specializations/data-science-python)
1. [Sklearn](https://scikit-learn.org)
1. [Feature Scaling with scikit-learn](http://benalexkeen.com/feature-scaling-with-scikit-learn/)
1. [https://docs.scipy.org/doc/numpy/user/quickstart.html](https://docs.scipy.org/doc/numpy/user/quickstart.html)
1. [https://pandas.pydata.org/](https://pandas.pydata.org/)
1. [https://www.stechies.com/numpy-indexing-slicing/](https://www.stechies.com/numpy-indexing-slicing/)
1. [python_pandas_dataframe](https://www.tutorialspoint.com/python_pandas/python_pandas_dataframe.htm)
###### [Go to top](#top)

**you may be interested have a look at it: [Course Home Page](https://www.kaggle.com/mjbahmani/10-steps-to-become-a-data-scientist)**
