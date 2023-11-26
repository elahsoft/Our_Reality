# Our_Reality
Using OLS to fit a data to a Linear Regression Model and find four best ideal functions to represent the models. And map the four ideal functions to the test data based on the criteria that the existing maximum deviation of the computed regression is not greater than the largest deviation between the selected ideal functions and the training dataset. The training dataset, test dataset with ideal functions mapped and their deviations incorporated, the ideal function dataset were written to a database  our_reality.db.

## Getting Started
To get started, the following softwares have to be installed:
1. Anaconda package and environment manager.
2. Visual Studio Code
After that, copy the anaconda folder created in the active user directory on your pc, and add it to the environmental variables path of your OS, so that the anaconda commands for creating an environment and installing the needed python packages for the project are available to you at the OS command prompt.
### Creating same Environment as development environment
1. Create your environment using the command: conda create --name <environment-name>
2. Import the development environment packages saved in the file req.txt into the created environment using the command: conda create -n <environment-name> --file req.txt

### Running Tests
1. Clone the project repository into the folder you want to house the project.
2. Open command prompt and navigate to the folder bearing the project
3. Run the command: python -m unittest discover -s . -p "*.py" to run the tests.

### Running Jupyter Notebook
1. Clone the project repository into the folder you want to house the project.
2. Open command prompt and navigate to the folder bearing the project
3. Activate the environment for the project
4. Type the command: jupyter notebook' at the command prompt
5. The directory of the project opens, select the our_reality.ipynb file.
6. Click on: Kernel and click on: Restart and Run all.

## Built With
* Visual Studio Code
* Anaconda
* Python Libraries as listed in the file req.txt

## Authors

**OPARA FEBECHUKWU CHINONYEREM**


## Institution
**International University of Applied Sciences, Germany


## Acknowledgments

* International University of Applied Sciences, Germany
* PROF. DR. CROITORU, COSMINA