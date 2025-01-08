# ETL process

## Requirements
- Python 3.11 must be installed in your machine
- An IDE must be installed (PyCharm or Visual Studio)

## Setup & Configurations
*Please note all the below commands are valid only for Windows OS, except for pip commands*

Open the command prompt in your machine and type the below commands

- Install python to your machine (Windows, Linux and Unix)
  - https://www.python.org/downloads/
- Check if Python is installed correctly
```bash
python --version
```
- Check if pip is installed correctly
```bash
pip -V
```
- If pip is not installed.Please go through the below commands.
```bash
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```

- Install virtualenv
```bash
pip install virtualenv
```

- Go to the project directory and create the virtual environment.
```bash
virtualenv env
```

- Activate virtualenv
```bash
env\Scripts\activate
```

*If you want to exit from the virtual environment you can use the below command*
- Deactivate the virtual environment
```bash
deactivate
```

- Install the Poetry lib (**At this time venv need to be in virtual environment active state**)

```bash
env\Scripts\activate
pip install poetry
```  

- Install the dependencies using poetry
  
```bash
poetry install
```
> Poetry will handle all the libraries and their versions, required for the project. You need to specify the required library names and the respective version in the pyproject.toml file, which is in the project folder. This file will be used to install all the mentioned libraries and packeages for the project. Once they are installed, poetry.lock file in the project folder will contain all the details of installed libraries.

- Run the ETL Process
  
```bash
python etl.py
```