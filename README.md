DaFy is a Python software package that contains a bunch of PyQt5-based APPs for processing synchrotron X-ray data. 
Refer to https://github.com/jackey-qiu/DaFy/wiki for more details.

In pydafy project, the constituent software components in original DaFy repo have been restructured/cleaned up/optimized to make it scarable and easier to understand.

**To install and use pydafy:**

1. setup a virtual env best using conda: eg `conda create --name dafy python==3.9.13 --no-default-packages`. Note python3.9.13 has been tested, but other higher python versions are not tested and may cause some unexpected problems when installing dependency packages. 
2. `activate dafy` to switch the newly created env
3. Download the source code, `git clone https://github.com/jackey-qiu/pydafy.git` and cd to pydafy root folder. 
4. Then use `pip install .` command to install dafy and all dependent pyhton packages to the python site-packages in the current env.
5. To launch it simply use command: `dafy --app CMD`, where CMD can be one of the items in this list [ubmate, ctr, superrod, archiver, xrv, viewer]. eg. `dafy --app superrod` will launch superrod app.
