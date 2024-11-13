



### Q- What is the difference between package and library ?
- **A Package:**
  - A collection or set of python files where each files corresponds to a module. 
  - This modules are organized together and are installed as a unit.
  - It is possible that one package contains several modules or even several sub-packages and other files such as documentation or configuration files.
  - Installation can be done using `pip` or `conda` in case you are using `Anaconda`.
  - A package usually has an __init__.py file (in Python) that marks the directory as a package.
  - Exampes: `numpy`, `pandas`
### Q- What are the common python libraries?

- **pyenv:**
  - It is a simple tool to manage multiple versions of Python on the same machine and for various development environments. 
  - It simplify and facilitate switching between different Python versions, on a single machine, without interfering with the system Python installation.
  - Code examples :
    - `pyenv install 3.9.2` : Install version 3.9.2 of Python
    - `pyenv global 3.9.2`: make this python version global
    - `pyenv local 3.8.6`: make this python version local but you must cd to your project folder.
    - `pyenv install --list` : list all existing python versions
