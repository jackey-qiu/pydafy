DaFy is a Python software package that contains a bunch of PyQt5-based APPs for processing synchrotron X-ray data. 
Refer to https://github.com/jackey-qiu/DaFy/wiki for more details.

In pydafy project, the constituent software components in original DaFy repo have been restructured/cleaned up/optimized to make it scalable, extendable, and easier to understand.

**To install and use pydafy:**

1. setup a virtual env best using conda: eg `conda create --name dafy python==3.9.13 --no-default-packages`. Note python3.9.13 has been tested, but other python versions are not tested and may cause some unexpected problems when installing dependency packages. 
2. `activate dafy` to switch the newly created env
3. Download the source code, `git clone https://github.com/jackey-qiu/pydafy.git` or download directly zipped file and cd or (unzip first in the case of zipped file and then cd) to pydafy root folder. 
4. Then use `pip install .` command to install dafy and all dependent pyhton packages to the python site-packages in the current env.
   This step is where you are most likely to get some trouble with, espectially when you are using macOS. If the dependency package installation fails at some point, find out which package is not installed successfully. Then you can manualy install that failed package using conda or homebrew package manager (macOS). Once succeed to install the package, run the `pip install .` command again to install the rest uninstalled packages, the installed packages will be ignored during installation when using pip install command.  
6. To launch it simply use command: `dafy --app CMD`, where CMD can be one of the items in this list [ubmate, ctr, superrod, archiver, xrv, viewer]. eg. `dafy --app superrod` will launch superrod app.
