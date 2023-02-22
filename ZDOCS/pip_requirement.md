# All pip requirements

First of all, python version must >= 3.8, then, use pip to install following packages:

## Group 1: Common
```
numpy scipy torch gym scikit-learn pygame 
```


## Group 2: Unreal-HMAP and Starcraft
```
lz4
(Note: our platform modifies ```smac``` to support fast switch between different Starcraft versions. Thus you must use our docker if you'd like to run ```smac```.)
```

## Group 3: Visual and Web Service
```
flask waitress colorama matplotlib ipykernel
``` 

## Group 4: Performance Boost
```
numba cython 
```

## Group 5: Functional
```
func_timeout commentjson PyYAML onedrivedownloader redis filelock scikit-fuzzy
```

## Group 6: Remote and management
```
paramiko psutil setproctitle sacred
```

## Install
``` 
pip install torch
pip install numpy scipy gym scikit-learn pygame lz4 flask waitress colorama matplotlib ipykernel numba cython func_timeout commentjson PyYAML onedrivedownloader redis filelock paramiko psutil setproctitle sacred
```

