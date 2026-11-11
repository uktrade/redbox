# Developer Setup Guide <!-- omit in toc -->

The developer setup guide for developing Redbox on your own machine.

## Table of Contents <!-- omit in toc -->

- [1. Set up Python Version Management](#1-set-up-python-version-management)
  - [Option 1: Using asdf (recommended if you're using other languages too)](#option-1-using-asdf-recommended-if-youre-using-other-languages-too)
  - [Option 2: Using pyenv (recommended for python only projects)](#option-2-using-pyenv-recommended-for-python-only-projects)
- [2. Set up direnv](#2-set-up-direnv)
- [3. Set up pre-commit hooks](#3-set-up-pre-commit-hooks)
- [4. Install Poetry](#4-install-poetry)
- [5. Install python packages](#5-install-python-packages)
- [6. Install NPM and node packages](#6-install-npm-and-node-packages)
- [7. Configure VSCode](#7-configure-vscode)
  - [VDE Port forwarding](#vde-port-forwarding)
  - [Python Interpreter](#python-interpreter)
- [8. Set environment variables](#8-set-environment-variables)
  - [Backend Profiles](#backend-profiles)
- [9. Set AWS credentials](#9-set-aws-credentials)
  - [Option 1: Set credentials with a command](#option-1-set-credentials-with-a-command)
  - [Option 2: Set credentials manually](#option-2-set-credentials-manually)
- [10. Set up a developer S3 bucket](#10-set-up-a-developer-s3-bucket)
- [11. Build service images](#11-build-service-images)
- [12. Running the service locally](#12-running-the-service-locally)
  - [One-time setup](#one-time-setup)
  - [Option 1: Running in Docker](#option-1-running-in-docker)
  - [Option 2: Running in Debug Mode](#option-2-running-in-debug-mode)
  - [Option 3: Hybrid approach](#option-3-hybrid-approach)
- [🏁 Checkpoint 1: Accessing the service](#-checkpoint-1-accessing-the-service)
- [13. Accessing the Admin UI - Superuser status](#13-accessing-the-admin-ui---superuser-status)
- [14. Setting up the chat llm backend](#14-setting-up-the-chat-llm-backend)
- [🏁 Checkpoint 2: LLM functionality](#-checkpoint-2-llm-functionality)
- [Next steps](#next-steps)


<hr>

## 1. Set up Python Version Management

To ensure everyone uses the same Python version, follow one of the two options below depending on your preference or existing setup

### Option 1: Using asdf (recommended if you're using other languages too)

Note: If you alreday have pyenv setup and would like to switch, ensure you've commented all pyenv initialization lines in your `~/.zshrc` or `~/.bashrc`. You may also need to restart your terminal and remove any existing pyenv venvs (Check `which python` and `poetry env info`. The base should point to the asdf-installed python).

1. Install asdf. Installation instructions are [here](https://asdf-vm.com/guide/getting-started.html).

2. Install the python plugin:
    ```bash
    asdf plugin add python
    ```
3. Install the required Python version:
    ```bash
    asdf install python
    ```

4. Tell Poetry to use this Python. Because asdf uses shims, Poetry needs to be explicitly told what Python to use. From the project root, and each individual app directory (`./django_app` & `./redbox`), run:
    ```bash
    poetry env use $(asdf which python)
    ```

### Option 2: Using pyenv (recommended for python only projects)

1. Install pyenv. Installation instructions [here](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation)

2. Restart your terminal or run
    ```bash
    source ~/.zshrc/source ~/.bashrc
    ```

3. Install the required python version. Check the project's `.tool-versions` or `pyproject.toml`. Then from the projet root, run:
   ```bash
    pyenv install $(awk '/^python / {print $2}' .tool-versions)
    pyenv local $(awk '/^python / {print $2}' .tool-versions)
    ```

    This sets the local version in the project repository. Poetry will automatically detect the pyenv-managed Python version.


### Check python verison

Run this command to verify successful Python configuration:

```bash
python --version
# Should output the correct Python version
```

## 2. Set up direnv

We recommend installing direnv to simplify the running of docker commands. Without it you will need to prefix each docker command with `DOCKER_DEFAULT_PLATFORM=linux/amd64`.

To install:

```bash
brew install direnv
```

Then add the following to your `~/.zshrc` or appropriate shell as seen [here](https://direnv.net/docs/hook.html).

```bash
eval "$(direnv hook zsh)"
plugins=(git direnv)
```

## 3. Set up pre-commit hooks
We use [pre-commit](https://pre-commit.com) for pre-commit hooks.
Run: `pip install pre-commit`
Then:  `pre-commit install`


## 4. Install Poetry

Poetry is not installed automatically when installing Python with `asdf` or `pyenv`, so you must install it on your host.

1. We recommend using Poetry's official installer:

    ```
    curl -sSL https://install.python-poetry.org | python3 -
    ```

2. After installation, ensure the Poetry binary is on your PATH. You may want to add the following to the end of your `~/.zshrc` or `~/.bashrc` (dependent on which shell you use). This means the binary will be loaded whenever you open a new terminal.

    ```
    export PATH="$HOME/.local/bin:$PATH"
    ```

3. Run these commands to verify successful installation:

    ```bash
    python --version
    # Should output the correct Python version

    poetry run python --version
    # Should also output the correct Python version

    # From each application root:
    poetry env info
    # Should show correct path to virtualenv using that Python version
    ```

## 5. Install python packages

We use [poetry](https://python-poetry.org/) to manage our python packages. There are 4 `pyproject.toml`s that define packages used by separate parts of the service.

Run the command `poetry install` in the following directories to install all dependencies:

|Directory|Dependencies for...|
|-|-|
|[. (project root)](https://github.com/uktrade/redbox/blob/main/pyproject.toml)|Integration tests, QA, and docs|
|[./django_app](https://github.com/uktrade/redbox/blob/main/django_app/pyproject.toml)|Django webserver and background worker|
|[./redbox](https://github.com/uktrade/redbox/blob/main/redbox/pyproject.toml)|Core AI package|
|[./notebooks](https://github.com/uktrade/redbox/blob/main/notebooks/pyproject.toml)|Jupyter notebooks|


## 6. Install NPM and node packages

Node Package Manager (NPM) manages node packages (javascript dependencies) used by the frontend.

1. Run `brew install node` to install NPM on your host (VDE).
2. Install node packages. Go to `./django_app/frontend` & run `npm i`.

**NB: If you work across several repos that use Node, we recommend using [`nvm`](https://github.com/nvm-sh/nvm?tab=readme-ov-file#installing-and-updating) (Node version manager) to manage node versions.**

To install `nvm`:

1. `curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash`
2. Restart your terminal.
3. Install Node:  
    ```console
    nvm install node
    nvm use --lts
    ```
4. Verify installation:  
    `node --version`

## 7. Configure VSCode

VSCode is our preferred IDE. The `.vscode/` directory is used for defining project-wide VSCode IDE settings.

### VDE Port forwarding

As the project runs on a VDE, you must ensure you have correctly setup port forwarding using the [instructions](https://uktrade.atlassian.net/wiki/spaces/AW/pages/5037457435/Port+forwarding+problems). At the time of writing, the ports that need to be forwarded are:
- 8001
- 9002
- 9003
- 8080
- 5432
- 5601
- 9200
- 5678

### Python Interpreter

Ensure your python interpreter is set to the root venv Python binary (should be `./venv/bin/python` or `./.venv/bin/python`).

1. Open the command pallette by pressing `Cmd + Shift + P` (Mac)
2. Select `Python: Select Interpreter`
3. Select `Enter interpreter path`, then `Find`, and navigate to `./venv/bin/python` or `./.venv/bin.python`. Select that Python executable.

Once the correct interpreter is selected it should display the `pyproject.toml` name `ie. Python 3.12.7 (redbox-root-py3.12)`. Also, opening any new terminals in VSCode will automatically activate that environment (ie. `source venv/bin/activate` or `source .venv/bin/activate`).


## 8. Set environment variables

We use `.env` files to populate the environment variables for local development. When cloning the repository the file `.env.example` will be populated.

1. Copy `.env.example` into a new `.env` file at repo root:  
`cp .env.example .env`
2. Set the relevant environment variables. The .env file indicates which variables you need to populate manually.


It is best to leave hostnames out of the .env file. These are then set manually by vscode tasks or pulled from a deployment .env like .env.test/.env.integration


### Backend Profiles

Redbox can use different backends for chat and embeddings. These are set using env vars. The defaults are currently to use Bedrock for both chat and embeddings but other providers can be used (and pointed to their relevant compliant local service).
The relevant env vars for overriding to use bedrock's titan model for embeddings are:

- `EMBEDDING_BACKEND` - usually `amazon.titan-embed-text-v2:0`

**`.env` and `.aws/credentials` are in `.gitignore` and should not be committed to git**


## 9. Set AWS credentials

**NB: Your AWS credentials need to be updated approximately daily as they reset regularly.**

### Option 1: Set credentials with a command

1. Run the following command to connect to AWS:  
  `aws configure sso`
2. Run the `aws-login` script in the project root to get AWS credentials for service tasks:  
    `./aws-login.sh`
3. You should now have a directory at `./.aws` containing files `credentials` & `config`. `credentials` shouild contain a `[default]` profile with the following values populated:  
    - `AWS_ACCESS_KEY`
    - `AWS_SECRET_ACCESS_KEY`
    - `AWS_SESSION_TOKEN`

### Option 2: Set credentials manually

1. Copy `.aws/credentials.example` into a new `.aws/credentials` file:  
`cp .aws/credentials.example .aws/credentials`
2. In `.aws/credentials` under profile `[default]` set the following:  
    - `AWS_ACCESS_KEY`
    - `AWS_SECRET_ACCESS_KEY`
    - `AWS_SESSION_TOKEN`
    - `AWS_CREDENTIAL_EXPIRATION` - default 30
3. Also create `.aws/config`:
`cp .aws/config.example .aws/config`


## 10. Set up a developer S3 bucket
Redbox uses s3 buckets for storage, even in local environments. To set this up, you will need to create your own bucket in the redbox AWS development account, the aws cli command for which you can find below:
```
aws s3api create-bucket \
  --profile <your-redbox-dev-aws-profile> \
  --bucket redbox-developer-bucket-<your-name> \
  --region eu-west-2 \
  --create-bucket-configuration LocationConstraint=eu-west-2
```
Then you will need to set BUCKET_NAME in the .env file to your bucket name: `redbox-developer-bucket-<your-name>`


## 11. Build service images

From project root, run:  
`make build`  
or  
`docker compose build`

Build takes a long time (20-30 minutes) so this is an excellent task to run before a meeting or break.


## 12. Running the service locally

The service can be run locally in several different ways, each with different benefits. We're working on a single containerised local run process that provides the benefits of all three approaches.

### One-time setup

The first time you run the service, run this command to start the project and perform a one-time setup. It runs a django loaddata command to install required models. This only needs to be done once.

```bash
make setup
```


### Option 1: Running in Docker

This process runs all service components in docker (apart from s3).

**Benefits:** Simple run process analogous to production.  
**Disadvanatages:** No hot reloading. The django app / worker need to be rebuilt if modified, which is time-consuming.

1. To run the service fully in docker run this command from project root:  
`docker compose up -d --wait`  
or  
`make run`
2. After a few minutes you should be able to access the service at [http://localhost:8080](http://localhost:8080)
3.  To stop the process run `docker compose down`.  
Remember to do this. If you keep the service running it can generate extensive logs which consume a lot of disk space.

### Option 2: Running in Debug Mode

This process uses the VS Code Debugger. The django app & worker run on the host machine (VDE), with all other components run in docker (apart from s3).

**Benefits:** Hot reloading. Allows use of the VS Code debugger.

1. Launch the service in debug mode.  
In the Run and Debug panel, use the dropdown to select 'Full Stack Dev (Frontend + Django), then hit the ▶️ button:
2. Launch QCluster.  
    In VS Code open the Command Pallette (ctrl+shift+p).  
    Run the command `Tasks: Run Task`.  
    Select task `Django: QCluster`.
3. After a few minutes you should be able to access the service at [http://localhost:8080](http://localhost:8080)
4. To stop the process hit the ⏹️ button in the Run and Debug panel.


### Option 3: Hybrid approach

This process runs the django app & worker on the host machine (VDE), with all other components running in docker (apart from s3), but it doesn't leverage the VS Code Debug tool or the associated run script.

**Benefits:** Hot reloading. Low complexity.

1. Start all services in docker apart from django_app and worker. In repo root, run:  
    `docker compose up -d --wait db minio opensearch sso`
2. From ./django_app, run:  
`DJANGO_LOG_LEVEL=WARNING poetry run python manage.py qcluster`
3. From ./django_app, run:  
`DJANGO_LOG_LEVEL=WARNING poetry run python manage.py runserver 8080`
4. After a few minutes you should be able to access the service at [http://localhost:8080](http://localhost:8080)



## 🏁 Checkpoint 1: Accessing the service

With the service set up and running you should be able to perform the following actions:
- Access the service at [`http://localhost:8080`](http://localhost:8080)
- Log in and out using the buttons/links in the UI

LLM functionality won't work yet. We'll set that up next.


## 13. Accessing the Admin UI - Superuser status

Assist has an admin UI at [http://localhost:8080/admin](http://localhost:8080/admin). However you can't initially access it. You need to give yourself superuser priveleges. You only need to do this once unless you delete your local database.

1. Visit [http://localhost:8080/admin](http://localhost:8080/admin). You won't be able to log in, but this will add your user to a local database so that we can give it superuser priveleges.
2. In terminal from repo root run `make superuser`.
3. When asked to enter a user account, enter `vyvyan.holland-20a0353f@id.mock-sso`. This account is used for mock SSO.
4. After finishng the `make superuser` dialogue visit [http://localhost:8080/admin](http://localhost:8080/admin) again. You should be shown the Assist Admin UI.


## 14. Setting up the chat llm backend

1. Get the ID of Assist's current default model. At time of writing it's `anthropic.claude-sonnet-4-6` but this changes over time.  
    To get the current default model:
   - Ask the team
   - If you have access, [go to  the llm settings in the dev deployment's admin UI](https://dev.assist.uktrade.digital/admin/redbox_core/chatllmbackend) and check which model is marked as `IS DEFAULT ✅`.

2. Go to http://localhost:8080/admin/redbox_core/chatllmbackend/.  
  Create a new chat llm backend with the following:
    ```
    Name: <the current default model's ID>  
    Provider: Bedrock  
    Is default: True  
    Enabled: True  
    ```

3. Go to http://localhost:8080/admin/redbox_core/aisettings/.  
    Check that the default settings uses the chat backend you just created.  
    Hit save.


## 🏁 Checkpoint 2: LLM functionality

You should now be able to perform the following actions:

- Access the service at http://localhost:8080
- Log in and out using the buttons/links in the UI
- Submit a query using the chat dialogue. The agent should give a reasonable response.
- Upload a file using the chat dialogue. Processing can take some time, expecially for large files, so try this out using a small .txt file.
- Submit a query with an uploaded a file using the chat dialogue. The agent should be able to access information in the file.
- Access the admin dashboard at http://localhost:8080/admin



---

## Next steps

You now have Assist running. Next you may want to set up some tools, and create knowledge bases for those tools. You can do that now, or you can set up tools as and when you need them. Instructions for this are in confluence.

<!-- Up to here.

Next steps:
1. Review this file.
2. Delete remaining obsolete content. -->
