# Submissions of gpumonkeys

This repository contains baseline submissions and a template for the submission of your software to the TIRA platform. The TIRA platform is used to evaluate your software on a set of datasets. The evaluation is done in a controlled environment to ensure reproducibility and comparability of the results.

We recommend that you work either in Github Codespaces or using [dev containers with Docker](https://code.visualstudio.com/docs/devcontainers/containers). Github Codespaces are an easy option to start in a few minutes (free tier of 130 compute hours per month), whereas dev container with Docker might be interesting if you want to put a bit more focus on technical/deployment details.

## Repository Structure

Each submission to a task should be self-contained in a directory within this repository. See the `authorship-verification-trivial` directory for an example. We have added an authorship-verification-submission directory for you to start developing your submission for the first task.

The `authorship-verification-trivial` example contains a single python file `authorship_verification_trivial.py` that loads the data and makes a prediction. TIRA handles loading the input datasets and saving the predictions into the correct location. The example also contains a `Dockerfile` that specifies how your submission is built and run when submitted to TIRA. In the simple example, it adds the python file to the container and runs it. Anything your submission needs (e.g., a saved model or other persited data) to run should be added to the `Dockerfile`.

A self-contained submission can then be submitted to TIRA using GitHub actions. See the [Submitting Your Software](#submitting-your-software) section for more details.

## Developing your Software Submission

You have two options for developing your submission. Either you can develop remotely in Github Codespaces or locally in a dev container.

### Developing in Github Codespaces

Click the `Code` button at the top of the repository and select `Open with Codespaces`. This will open a new Codespace with the repository loaded. You can then start developing your submission in the Codespace.

Additional information on how to use Github Codespaces can be found [here](https://docs.github.com/en/codespaces/).

### Developing in Dev Containers

A dev container (please find a suitable installation instruction [here](https://code.visualstudio.com/docs/devcontainers/containers)) allows you to directly work in the prepared Docker container so that you do not have to install the dependencies (which can sometimes be a bit tricky).

To develop with dev containers, please:

- Install [VS Code](https://code.visualstudio.com/download) and [Docker](https://docs.docker.com/engine/install/) on your machine
- Install the [Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) in VS Code
- Clone this repository: `git clone ...`
- Open the repository with VS Code (it should ask you to open the repository in a dev container)

## Submitting Your Software

Run the github action to submit your software. Select the Action tab and then select `Upload Docker Software to TIRA` from the list of workflows on the left. Select the `Run workflow` button input the directory of the submission you want to submit. Then select the dataset the submission should be run on and click `Run workflow`. The action will build the docker container, test everything works, and then upload the container to TIRA.

### Persisting Models/Data

Your submission will not have access to the training or validation data when running on Tira. Therefore, anything training you need to do should be done locally and the model persisted in the container. See the `authorship-verification-bayes` directory for an simple example of training a model using scikit learn, saving it to disk, and adding it to the submitted Docker image.

## Running your Software

After sucessfully submitting your software, it will be avaiable on TIRA. Select `Submit` and then the `Docker` tab. Your software submission will be listed under the `Choose sofware ...` dropdown menu (the names are generated by docker, check the output of the github action to see which name corresponds to which submission). Select your submission and then select the resources (small will usually suffice) and dataset you want to run it on. Click `Run` to run your software on the dataset.

You can check the progress of your submission by unfolding the `Running Processes` tab at the top.

## Additional Hints

Some additional advanced hints for developing your submission:

### Additional dependencies

If you require additional depenedencies, you can add them to the `Dockerfile` in the `dev-container` directory. Next, build the container and push it to Docker Hub. Then change the image in the `.devcontainer.json` file to point to your image.

### Running/Testing your submission locally

You can run it directly via (please install `tira` via `pip3 install tira`, Docker, and Python >= 3.7 on your machine) where you replace `{dataset-name}` with the name of the dataset and `{image-name}` with the name of the image you want to run:

```bash
tira-run \
    --input-dataset {dataset-name} \
    --image {image-name}
```
