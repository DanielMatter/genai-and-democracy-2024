# GenAI and Democracy Seminar

This repository accompanies the GenerativeAI and Democracy seminar at HfP / TUM. It provides useful material and links, as well as the boilerplate code to get started with your projects.

To get started with LLMs in general, please read the [Introduction on LLMs](INTRODUCTION-LLMs.md).


## Getting Started with the Project

This repository helps you set up the GenAI and Democracy Challenge. It provides boilerplate code, a small sample dataset, and some documentation to get you started.

**Please adjust ```user_config.py``` to your needs before starting.**


### Writing Your Code

You should focus on the three files ```user_setup.py```, ```user_preprocess.py```, and ```user_inference.py```. Each of these files contains stub functions where you can implement arbitrary code. **You cannot change the function signatures and return values**. 
The functions are called by the evaluation script, and changing the signatures will result in a runtime error.

### Running Your Code

Running your code will be done in a Docker container with a specific environment and startup sequence.

#### Environment
The Docker container will already contain ```python==3.11.0```, ```numpy==1.24.0```, ```torch==2.0.0```, and ```transformers==4.12.0```. We recommend setting up your local environment with these versions to avoid discrepancies.
We further recommend using as little additional dependencies as possible to avoid conflicts. However, you can add dependencies to the ```requirements.txt``` file.

#### Startup Sequence
The Docker container will run the following commands in the following order:
1. ```python user_setup.py```. If this script returns a non-zero exit code, the evaluation script will stop.
2. ```python user_preprocess.py -input "[FILE1];[FILE2];..."```. This script will be called with the path to all relevant input files. The script should preprocess the data and save it to a file as given in ```preprocessed-file.json```.
3. ```python user_inference.py -user_query "[USER QUERY]" -output_file "[OUTPUT_FILE]"```. This script will be called with a user query and the path to an output file. The script should generate a response to the user query and save it to the output file. You should access your previously saved preprocessed data in this script to respond to the user query.


### Testing Your Code

You can test each part of your pipeline individually.
To do so, you can call ```python test.py -part [PART]``` where ```[PART]``` is one of ```setup```, ```preprocess```, or ```inference```.
Note that this will only test if all data is loaded and stored correctly, but not how well your model performs.

