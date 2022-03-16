# [readme]pandas3d

## install
```bash
pip install git+https://github.com/eycjur/pandas3d
poetry add git+https://github.com/eycjur/pandas3d.git#main
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
