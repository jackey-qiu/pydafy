DaFy is a Python software package that contains a bunch of PyQt5-based APPs for processing synchrotron X-ray data. 
Refer to https://github.com/jackey-qiu/DaFy/wiki for more details.

In pydafy project, the constituent software components in original DaFy repo have been restructured/cleaned up/optimized to make it scarable and easier to understand.

**To install and use pydafy:**

1. Download the source code, and cd to pydafy root folder. 
2. Then use `pip install .` command to install it to the current env python site-packages.
3. To launch it simplying use command: `dafy --app CMD`, where CMD can be one of the items in this list [ubmate, ctr, superrod, archiver, xrv, viewer]. eg. `dafy --app superrod` will launch superrod app.
