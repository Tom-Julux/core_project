# Installation

## System Requirements
The core-tool requires a system with the following minimum specifications:
- Operating System: Linux or Windows (macOS support is limited)

## Clone the Repository
Start by cloning the repository containing the core-tool and its plugins.

```bash
git clone
cd core-tool
```

## Decide on usage method
The core-tool can be used either directly as an desktop application or via as a web application, via [xpra](https://xpra.org/). 
**For local usage, the desktop application is recommended.** For remote usage, the web application is recommended, but usage via a remote desktop application (e.g. VNC) is also possible. 

> Both usage methods only differ in the final command to start the application. The local usage is described as default and the remote usage in boxes like this one.

## Docker

The core-tool can be run using Docker. This is the recommended way to use the tool on a cluster or if you want to avoid installing dependencies on your local machine. Note that the Docker image is quite large (several GBs) due to the inclusion of multiple machine learning models and libraries.

For development or usage on a local machine, consider using the pip installation method below.

> Tip: MacOS users should use the pip installation method, as Docker support for macOS is limited.

During the build of the container read about the [fundamental concepts]() of the tool or proceed to the [first steps guide]().

## Local installation via pip

## (Optional) Create a Virtual Environment

## Next Step: Obtain Models

After installation, you need to obtain the machine learning models used by the core-tool. Follow the instructions in the [model download guide](../models/download.md) to download and set up the models