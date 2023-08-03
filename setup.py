from setuptools import setup, find_packages
setup(
    name = 'dafy',
    version= '0.1.0',
    description='data analysis factory for pocessing surface x ray diffraction data',
    author='Canrong Qiu (Jackey)',
    author_email='canrong.qiu@desy.de',
    url='https://github.com/jackey-qiu/Library_Manager',
    classifiers=['Topic :: x ray data analysis',
                 'Programming Language :: Python'],
    license='MIT',
    install_requires=['PyQt5','pyqtgraph','qdarkstyle','pymongo','python-dotenv','pandas','dnspython'],
    packages=find_packages(),
    package_data={'':['*.ui','*.ini','*.dat', '*.cif'],'dafy.projects.xrv':['icons/*.*','config/config_file_XRV_standard.ini','ui/*.ui'], 'dafy.projects.superrod':['icons/*.*'],\
                  'dafy.projects.ubmate':['icons/*.*'],'dafy.projects.ctr':['icons/*.*','config/*.ini','ui/*.ui'],'dafy.projects.viewer':['icons/*.*','config/*.ini','ui/*.ui'],\
                  'dafy.resources.cif':['*.cif'], 'dafy.resources.cif':['Au/*.*','Co/*.*','Cu/*.*','Fe/*.*'],\
                  'dafy.resources':['cif/*.cif','batchfile/*/*.*','example/*/*','example/*/*/*'],\
                  'dafy.projects.archiver.resources':['icons/*.*','private/*.*','templates/*.ini','optics/*.gz'],\
                  'dafy.projects.superrod.core.models':['databases/f1f2_cxro/*','databases/*.dat']},
    entry_points = {
        'console_scripts' : [
            'dafy = dafy.bin.dafy_launcher:dispatcher'
        ],
    }
)