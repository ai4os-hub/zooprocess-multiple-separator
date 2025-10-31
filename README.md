# zooprocess-multiple-separator
[![Build Status](https://jenkins.services.ai4os.eu/buildStatus/icon?job=AI4OS-hub/zooprocess-multiple-separator/main)](https://jenkins.services.ai4os.eu/job/AI4OS-hub/job/zooprocess-multiple-separator/job/main/)

This AI module processes images produced by [ZooProcess](https://sites.google.com/view/piqv/softwares/flowcamzooscan) that may contain multiple objects and separates the objects by drawing lines between them. This allows the rest of the processing to work on individual objects, which is necessary for their classification and measurements.

The segmentation is based on Mask2Former, in panoptic mode, but the resulting detections are further processed to match the original objects to the pixel and draw the separation lines.

This module was developed as part of the [iMagine](https://www.imagine-ai.eu) project, by the [Laboratoire d'Océanographie de Villefranche (LOV)](http://lov.imev-mer.fr/) in partnership with the company [FOTONOWER](http://fotonower.com/).


## Run the module's API

First download and install the package and its dependencies. It is good practice to perform this installation in a virtual environment (see the documentation for [Python native venvs](https://docs.python.org/3/library/venv.html) or [Conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#)). The dependencies are tested with Python 3.12 and are installed with `pip` so, to create a compatible environment with conda, one would write:

```bash
conda env create --name=zooprocess_multiple python=3.12
# this installs pip too
conda activate zooprocess_multiple
```

Then

```bash
git clone https://github.com/ai4os-hub/zooprocess-multiple-separator
cd zooprocess-multiple-separator
pip install -e .
```

Fetch the model weights from the first GitHub release and move them in the appropriate location

```bash
wget https://github.com/ai4os-hub/zooprocess-multiple-separator/releases/download/v1.0.0/learn_plankton_pano_plus5000_8epoch.zip
mv learn_plankton_pano_plus5000_8epoch.zip models/
```

Run [DEEPaaS](https://github.com/ai4os/DEEPaaS):

```bash
deepaas-run --listen-ip 0.0.0.0 --listen-port 5001 --model-name zooprocess_multiple_separator
```

NB: the port is set to 5001 rather than the default 5000 so that it does not conflict with the module [zooprocess-multiple-classifier](https://github.com/ai4os-hub/zooprocess-multiple-classifier) that you would probably also be using.

Then browse to <http://localhost:5001> and you should get a simple message as a response, verifying that everything works as intended.
Finally, browse to <http://localhost:5001/api> to get access to the graphical interface and documentation of the DEEPaaS API for this model.


## Docker images

Once a version is pushed to the main branch, it should be built automatically through [AI4OS Jenkins service](https://jenkins.services.ai4os.eu/job/AI4OS-hub/job/zooprocess-multiple-separator/) and become available from [dockerhub](https://hub.docker.com/r/ai4oshub/zooprocess-multiple-separator/tags).

Getting the image works through
```bash
docker pull ai4oshub/zooprocess-multiple-separator:latest
```

When running the docker image, map port 5000 from inside the docker (NB: this is now 5000, which is the standard port, not 5001 like above) to a port on the machine running docker (e.g. 55001). Then replace `http://localhost:5001` above by the ip/name of the machine and the port you mapped 5000 to (55001 in the example above). As above, this uses port 55001 to avoid conflicts if you already run the docker for [zooprocess-multiple-classifier](https://github.com/ai4os-hub/zooprocess-multiple-classifier) on port 55000.

<img src="illustration_separator.png" style="width: 100%;" alt="illustration" />

## Project structure
```
│
├── Dockerfile             <- Describes main steps on integration of DEEPaaS API and
│                             zooprocess_multiple_separator application in one Docker image
│
├── Jenkinsfile            <- Describes basic Jenkins CI/CD pipeline (see .sqa/)
│
├── LICENSE                <- License file
│
├── README.md              <- The top-level README for developers using this project.
│
├── VERSION                <- zooprocess_multiple_separator version file
│
├── .sqa/                  <- CI/CD configuration files
│
├── zooprocess_multiple_separator    <- Source code for use in this project.
│   │
│   ├── __init__.py        <- Makes zooprocess_multiple_separator a Python module
│   │
│   ├── api.py             <- Main script for the integration with DEEPaaS API
│   |
│   ├── config.py          <- Configuration file to define Constants used across zooprocess_multiple_separator
│   │
│   └── misc.py            <- Misc functions that were helpful accross projects
│   │
│   └── utils.py           <- Contains the actual code to perform inference
│
├── data/                  <- Folder to store the data
│
├── models/                <- Folder to store models
│   
├── tests/                 <- Scripts to perfrom code testing
|
├── metadata.json          <- Metadata information propagated to the AI4OS Hub
│
├── pyproject.toml         <- a configuration file used by packaging tools, so zooprocess_multiple_separator
│                             can be imported or installed with  `pip install -e .`                             
│
├── requirements.txt       <- The requirements file for reproducing the analysis environment, i.e.
│                             contains a list of packages needed to make zooprocess_multiple_separator work
│
├── requirements-test.txt  <- The requirements file for running code tests (see tests/ directory)
│
└── tox.ini                <- Configuration file for the tox tool used for testing (see .sqa/)
```
