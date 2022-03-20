# [readme]pandas3d

## install
```bash
# pip
pip install git+https://github.com/eycjur/pandas3d

# poetry
poetry add git+https://github.com/eycjur/pandas3d.git#main

# colab
!pip install git+https://github.com/eycjur/pandas3d@0.1.0colab
```

## docs
[https://eycjur.github.io/pandas3d/](https://eycjur.github.io/pandas3d/)

## use
```python
import numpy as np
import pandas3d as pd3
array = np.arange(12).reshape(2, 3, 2)
gf = pd3.GridFrame(data=array, columns=['a', 'b'])
gf
# a
# [[ 0  2  4]
#  [ 6  8 10]]
# b
# [[ 1  3  5]
#  [ 7  9 11]]
# shape(2, 3, 2), dtype('int64')

gf["c"] = gf["a"] + gf["b"]
gf
# a
# [[ 0  2  4]
#  [ 6  8 10]]
# b
# [[ 1  3  5]
#  [ 7  9 11]]
# c
# [[ 1  5  9]
#  [13 17 21]]
# shape(2, 3, 3), dtype('int64')
```

## warning
- `gf.__values`は単にnumpy.ndarrayを保持しているだけのため、統一された型になります。
- `gf.loc`は実装していません。

## reference
- [【Python】自作モジュール内でloggingする](https://qiita.com/Esfahan/items/275b0f124369ccf8cf18)
