# Developer Setup Guide

The developer setup guide for developing Redbox on your own machine.

## Table of Contents

- [Developer Setup Guide](#developer-setup-guide)
  - [Table of Contents](#table-of-contents)
  - [Python Version Management Setup Guide \[\*Required\]](#python-version-management-setup-guide-required)
    - [Option 1: Using asdf (recommended if you're using other languages too)](#option-1-using-asdf-recommended-if-youre-using-other-languages-too)
      - [Step 1: Install asdf](#step-1-install-asdf)
      - [Step 2: Install the python plugin](#step-2-install-the-python-plugin)
      - [Step 3: Install the required Python version](#step-3-install-the-required-python-version)
      - [Step 4: Tell Poetry to use this Python](#step-4-tell-poetry-to-use-this-python)
    - [Option 2: Using pyenv (recommended for python only projects)](#option-2-using-pyenv-recommended-for-python-only-projects)
      - [Step 1: Install pyenv](#step-1-install-pyenv)
      - [Step 2: Install the required python version](#step-2-install-the-required-python-version)
  - [Install Poetry \[\*Required\]](#install-poetry-required)
  - [Install Project Dependencies with Poetry \[\*Required\]](#install-project-dependencies-with-poetry-required)
    - [Local Install](#local-install)
    - [Verifying Setup](#verifying-setup)
  - [Setup VSCode](#setup-vscode)
    - [Python Interpreter \[\*Required\]](#python-interpreter-required)
      - [Verify Setup](#verify-setup)
    - [Code Workspaces \[Optional\]](#code-workspaces-optional)
    - [Devcontainer \[Optional\]](#devcontainer-optional)
  - [Setting environment variables](#setting-environment-variables)
    - [Backend Profiles](#backend-profiles)
  - [Running the Project Locally](#running-the-project-locally)
    - [Building and running the project - Docker \& Make](#building-and-running-the-project---docker--make)
    - [Building and running the project - VSCode Debugger](#building-and-running-the-project---vscode-debugger)
      - [1. Set VSCode Python Interpreter to `django_app/`](#1-set-vscode-python-interpreter-to-django_app)
      - [2. Run the main app](#2-run-the-main-app)
      - [3. Run the worker](#3-run-the-worker)
    - [How to run tests](#how-to-run-tests)
      - [Integration tests](#integration-tests)
    - [Logging in to Redbox Locally as a superuser](#logging-in-to-redbox-locally-as-a-superuser)
    - [Setting up the chat llm backend](#setting-up-the-chat-llm-backend)
    - [Setting up AWS credentials](#setting-up-aws-credentials)
    - [Running Redbox in a notebook](#running-redbox-in-a-notebook)
      - [Configuring the notebooks kernel in vscode](#configuring-the-notebooks-kernel-in-vscode)
      - [Configuring notebook environment variables](#configuring-notebook-environment-variables)
  - [Git](#git)
    - [Branch Naming](#branch-naming)
    - [Pre-commit hooks](#pre-commit-hooks)
  - [LLM evaluation](#llm-evaluation)
    - [Installing Node and `elasticsearch-dump`](#installing-node-and-elasticsearch-dump)
    - [Dumping data from Elasticsearch](#dumping-data-from-elasticsearch)
    - [Loading data to Elasticsearch](#loading-data-to-elasticsearch)
  - [Iconography](#iconography)

<hr>

## Python Version Management Setup Guide [*Required]

To ensure everyone uses the same Python version, follow one of the two options below depending on your preference or existing setup

### Option 1: Using asdf (recommended if you're using other languages too)

Note: If you alreday have pyenv setup and would like to switch, ensure you've commented all pyenv initialization lines in your `~/.zshrc` or `~/.bashrc`. You may also need to restart your terminal and remove any existing pyenv venvs (Check `which python` and `poetry env info`. The base should point to the asdf-installed python).

#### Step 1: Install asdf

Installation instructions [here](https://asdf-vm.com/guide/getting-started.html)

#### Step 2: Install the python plugin

```bash
asdf plugin add python
```

#### Step 3: Install the required Python version

From project root:

```bash
asdf install python
```

This installs and sets the local Python version for the project.

#### Step 4: Tell Poetry to use this Python

Because asdf uses shims, Poetry needs to be explicitly told what Python to use. From the project root and each individual app, run:

```bash
poetry env use $(asdf which python)
```

### Option 2: Using pyenv (recommended for python only projects)

#### Step 1: Install pyenv

Installation instructions [here](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation)

Restart your terminal or run

```bash
source ~/.zshrc/source ~/.bashrc
```

#### Step 2: Install the required python version

Check the projects .tool-versions or pyproject.toml. Then from the projet root, run:

```bash
pyenv install $(awk '/^python / {print $2}' .tool-versions)
pyenv local $(awk '/^python / {print $2}' .tool-versions)
```

This sets the local version in the project repository

Poetry will automatically detect the pyenv-managed Python version.

## Install Poetry [*Required]

Poetry is not installed automatically when installing Python with `asdf` or `pyenv`, so you must install once on your host machine.

1. We recommend using Poetry's official installer:

```
curl -sSL https://install.python-poetry.org | python3 -
```

2. After installation, ensure the Poetry binary is on your PATH. You may want to add the following to the end of your `~/.zshrc` or `~/.bashrc` (dependent on which shell you use). This means the binary will be loaded whenever you open a new terminal.

```
export PATH="$HOME/.local/bin:$PATH"
```

## Install Project Dependencies with Poetry [*Required]

Currently, we use [poetry](https://python-poetry.org/) to manage our python packages. There are 4 `pyproject.toml`s

- [redbox](https://github.com/i-dot-ai/redbox/blob/main/redbox/pyproject.toml) - core AI package
- [django-app](https://github.com/i-dot-ai/redbox/blob/main/django_app/pyproject.toml) - django webserver and background worker
- [root](https://github.com/i-dot-ai/redbox/blob/main/pyproject.toml) - Integration tests, QA, and docs
- [notebooks](https://github.com/uktrade/redbox/blob/main/notebooks/pyproject.toml) - Jupyter notebooks

### Local Install

Once Python has been configured and installed using either `pyenv` or `asdf`, and Poetry installed - from each applications root directory (`django_app`, `redbox`, `notebooks`), run the following:

```bash
poetry install
```

### Verifying Setup

Run these to confirm:

```bash
python --version
# Should output the correct Python version

poetry run python --version
# Should also output the correct Python version

# From each application root:
poetry env info
# Should show correct path to virtualenv using that Python version
```

## Setup VSCode

VSCode is the IDE of choice. The `.vscode/` directory is used for defining project-wide VSCode IDE settings.

### Python Interpreter [*Required]

Ensure your python interpreter is set to the root venv Python binary (should be `./venv/bin/python` or `./.venv/bin/python`).

#### Verify Setup

Once the correct interpreter is selected it should display the `pyproject.toml` name `ie. Python 3.12.7 (redbox-root-py3.12)`. Also, opening any new terminals in VSCode will automatically activate that environment (ie. `source venv/bin/activate` or `source .venv/bin/activate`).

### Code Workspaces [Optional]

To make use of the VSCode Workspaces setup open the workspace file `.vscode/redbox.code-workspace`. This will open the relevant services as roots in a single workspace. The recommended way to use this is:

- Create a venv in each of the main service directories (redbox, django-app) this should be in a directory called `venv`
- Configure each workspace directory to use it's own venv python interpreter. NB You may need to enter these manually when prompted as `./venv/bin/python`

The tests should then all load separately and use their own env.

### Devcontainer [Optional]

The devcontainer currently is not supported for project-wide dependency setup so it is generally recommended to do development on your host machine.

## Setting environment variables

We use `.env` files to populate the environment variables for local development. When cloning the repository the file `.env.example` will be populated.

To run the project:

- `cp .env.example .env`
- `cp .aws/credentials.example .aws/credentials`

Then set the relevant environment variables.

Typically this involves setting the following variables in .aws/credentials (after running `cp .aws/credentials.example .aws/credentials`):

- `AWS_ACCESS_KEY`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_SESSION_TOKEN`
- `AWS_CREDENTIAL_EXPIRATION` - default 30

It is best to leave hostnames out of the .env file. These are then set manually by vscode tasks or pulled from a deployment .env like .env.test/.env.integration

### Backend Profiles

Redbox can use different backends for chat and embeddings, which are used is controlled by env vars. The defaults are currently to use Bedrock for both chat and embeddings but other providers can be used (and pointed to their relevant compliant local service).
The relevant env vars for overriding to use bedrock's titan model for embeddings are:

- `EMBEDDING_BACKEND` - usually `amazon.titan-embed-text-v2:0`

**`.env` and `.aws/credentials` are in `.gitignore` and should not be committed to git**

## Running the Project Locally

How to run the project locally. This includes setting up AWS credentials.

### Building and running the project - Docker & Make

To view all the build commands, check the `Makefile` that can be found [here](https://github.com/i-dot-ai/redbox/blob/main/Makefile).

The project currently consists of multiple docker images needed to run the project in its entirety. If you only need a subsection of the project running, for example if you're only editing the django app, you can run a subset of the images. The images currently in the project are:

- `elasticsearch`
- `minio`
- `db`
- `django-app`
- `worker`

To build the images needed to run the project, use this command:

```bash
make build
```

or

```bash
docker compose build
```

Once those images have built, you can run them using:

```bash
make run
```

or

```bash
docker compose up
```

Some parts of the project can be run independently for development, for example the django application, which can be run with:

```bash
docker compose up django-app
```

Sometimes, you might have used too much memory from previous docker runs. Memory need to be flushed before running docker. You can use the following commands:

```bash
docker system prune --all --force

DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose build

DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose up

# The DOCKER_DEFAULT_PLATFORM=linux/amd64 is only needed for certain MAC OS. You can omit this by adding the variable to your .envrc file.
```

We recommend installing direnv to prevent having to specify DOCKER_DEFAULT_PLATFORM for each docker command. To install:

```bash
brew install direnv
```

Then add the following to your `~/.zshrc` or appropriate shell as seen [here](https://direnv.net/docs/hook.html).

```bash
eval "$(direnv hook zsh)"
plugins=(git direnv)
```

For any other commands available, check the `Makefile` [here](https://github.com/i-dot-ai/redbox/blob/main/Makefile).

### Building and running the project - VSCode Debugger

You can also choose to run the project with the VSCode Python Debugger, allowing you to create breakpoints in the code for programmatic inspection.

> [!WARNING]
> Please be aware this debugging implementation is relatively new and has some nuances due to deviations from the docker build configuration.

#### 1. Set VSCode Python Interpreter to `django_app/`

1. Select a python file to view selected interpreter
2. Set to `django_app/venv/bin/python` or `django_app/.venv/bin/python`
3. Interpeter should display as `(redbox-app-py3.12)`

#### 2. Run the main app

1. Go to `Run and Debug` tab on left side of VSCode window
2. Go to green play button dropdown and select `Full Stack Dev (Frontend + Django)`
3. Click play button - should spin up dependency containers, build frontend, and then run main app with python debugger

#### 3. Run the worker

1. Open command palette - CMD + Shift + P
2. Select `Tasks: Run Task`
3. Run `Django: QCluster`

### How to run tests

Tests are split into different commands based on the application the tests are for. For each application there is a separate `make` command to run those tests, these are:

For the django app:

```bash
make test-django
```

For the core AI:

```bash
make test-redbox
```

For integration tests:

```bash
make test-integration
```

#### Integration tests

There is a suite of playwright integration tests in the tests/ folder. There is a make command `test-integration` that will setup the local environment to be able to run the playwright tests, then execute the test suite.

The make command runs the entire suite in a headless configuration, however it is possible to run the playwright tests in debug mode using the playwright inspector. This command will start the playwright inspector tool: `make test-integration-debug`

### Logging in to Redbox Locally as a superuser

We'll need to create a superuser to log in to the Django admin page, to do this run the following steps:

1. Log into Redbox at http://localhost:8080/sign-in
2. Run `make superuser` in your terminal
3. Use the `vyvyan.holland-20a0353f@id.mock-sso` email that is generated by the mock-sso service

### Setting up the chat llm backend

Once the app is up and running, head to http://localhost:8080/admin/redbox_core/chatllmbackend/

Create a new chat llm backend with the following:

Name:

```bash
# Example:
anthropic.claude-3-sonnet-20240229-v1:0
```

_This may change over time, to get the correct ID, head to amazon bedrock in the aws console > Foundation Models > model catalog > Claude 3 Sonnet > Model ID_

Provider:

```bash
Bedrock
```

Is default:

```bash
True
```

Enabled:

```bash
True
```

Save and head to http://localhost:8080/admin/redbox_core/aisettings/

Ensure the default settings uses the chat backend you just created and hit save again.

Chat and document uploads should now work as expected.

### Setting up AWS credentials

To recieve responses from the LLM you will need to have access to redboc aws account (See another member of the team about requesting access).

To configure your aws profile, run the following command or manaully update your `~/.aws/config` file with assistance from another team member.

```bash
aws configure sso
```

_Note: If using a non-default profile name, (e.g. redbox), please make sure you create an `.envrc` file with the `AWS_PROFILE` value set. See .envrc.example_

Once access has been provided and credentials configured, run the aws-login script in the project root and follow the instructions on-screen to connect.

```bash
./aws-login.sh
```

Once authenticated you should have a `.aws` directory within the project root and notebooks app with a `credentials` file populated. This directory is added to the gitignore and should _NOT_ be commited.

_Note: This script should be run periodically (daily) as the credentials will expire relatively soon._

### Running Redbox in a notebook

There are a number of notebooks available, in various states of working! The Redbox core app is able to be created in a notebook and run to allow easy experiementation without the django side.
agent_experiments.ipynb shows this best currently.

#### Configuring the notebooks kernel in vscode

In order to run notebooks in vscode, you will need to use the virtualenv created by poetry within the notebooks directory. If this does not appear as an option, you may need to add the notebooks directory path to your vscode python settings:

1. Open vscode settings: `[cmd + ,]`
2. Search: `python.venvFolders`,
3. Add the path to `./redbox/notebooks`

You may also want to add the path for the other apps in order to select the correct interpreter during development.

#### Configuring notebook environment variables

Some notebooks may require specific environment variables to run. For non-sensetive variables that apply to all notebooks, add them to `.env.notebook` and override the root `.env` at the top of your notebook like so:

```bash
dotenv .env
dotenv -o ./.env.notebook
```

For sensetive environment variables, please create a seperate `notebooks/.env` file within the notebooks directory and add them there. You can then override the `.env` and `.env.notebook` in the same way.

```bash
dotenv .env
dotenv -o ./.env.notebook
dotenv -o ./.env
```

## Git

The workflows for using Git.

### Branch Naming

Consistent branch names help maintain a clean and predictable workflow. CI will fail if your branch does not follow conventions. Use the following prefixes:

- `feature/<name>` — New features or enhancements
- `chore/<name>` — Maintenance tasks that don’t affect functionality
- `bugfix/<name>` — Non-critical bug fixes
- `hotfix/<name>` — Urgent fixes for production issues
- `dependabot/<name>` — Automated dependency updates
- `security/<name>` — Security-related changes
- `revert<name>` — For when git creates a branch to revert a PR

### Pre-commit hooks

- Download and install [pre-commit](https://pre-commit.com) to benefit from pre-commit hooks
  - `pip install pre-commit`
  - `pre-commit install`

## LLM evaluation

Notebooks with some standard methods to evaluate the LLM can be found in the [notebooks/](../notebooks/) directory.

You may want to evaluate using versioned datasets in conjunction with a snapshot of the pre-embedded vector store.

We use [elasticsearch-dump](https://github.com/elasticsearch-dump/elasticsearch-dump) to save and load bulk data from the vector store.

### Installing Node and `elasticsearch-dump`

Install [Node and `npm`](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm) (Node package manager) if you don't already have them. We recommend using [`nvm`](https://github.com/nvm-sh/nvm?tab=readme-ov-file#installing-and-updating) (Node version manager) to do this.

If you're familiar with Node or use it regularly we recommend following your own processes or the tools' documentation. We endeavour to provide a quickstart here which will install `nvm`, Node, `npm` and `elasticsearch-dump` globally. This is generally not good practise.

To install `nvm`:

```console
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
```

Restart your terminal.

Install Node.

```console
nvm install node
nvm use --lts
```

Verify installation.

```console
node --version
```

Install `elasticsearch-dump` globally.

```console
npm install elasticdump -g
```

### Dumping data from Elasticsearch

The default indicex we want is `redbox-data-chunk`

Dump these to [data/elastic-dumps/](../data/elastic-dumps/) for saving or sharing.

```console
elasticdump \
  --input=http://localhost:9200/redbox-data-chunk \
  --output=./data/elastic-dumps/redbox-data-chunk.json \
  --type=data
```

### Loading data to Elasticsearch

If you've been provided with a dump from the vector store, add it to [data/elastic-dumps/](../data/elastic-dumps/). The below assumes the existance of `redbox-data-chunk.json` in that directory.

Consider dumping your existing indices if you don't want to have to reembed data you're working on.

Start the Elasticsearch service.

```console
docker compose up -d elasticsearch
```

Load data from your JSONs, or your own file.

```console
elasticdump \
  --input=./data/elastic-dumps/redbox-data-chunk.json \
  --output=http://localhost:9200/redbox-data-chunk \
  --type=data
```

If you're using this index in the frontend, you may want to upload the raw files to MinIO, though that's out of scope for this guide.

## Iconography

We currently use [Google icons](https://fonts.google.com/icons). When adding new icons, ensure the following customizations are made:

```
Weight: 300
Grade: 0
Optical Size: 24px
Style:
 - Material Symbols (new)
 - Rounded
```
