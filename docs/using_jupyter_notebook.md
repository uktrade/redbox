# Running Notebooks

## 1. Running Redbox in a notebook

There are a number of notebooks available, in various states of working! The Redbox core app is able to be created in a notebook and run to allow easy experiementation without the django side.
agent_experiments.ipynb shows this best currently.

### Configuring the notebooks kernel in vscode

In order to run notebooks in vscode, you will need to use the virtualenv created by poetry within the notebooks directory. If this does not appear as an option, you may need to add the notebooks directory path to your vscode python settings:

1. Open vscode settings: `[cmd + ,]`
2. Search: `python.venvFolders`,
3. Add the path to `./redbox/notebooks`

You may also want to add the path for the other apps in order to select the correct interpreter during development.

### Configuring notebook environment variables

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


## 2. Using notebooks for LLM evaluation

Notebooks with some standard methods to evaluate the LLM can be found in the [notebooks/](../notebooks/) directory.

You may want to evaluate using versioned datasets in conjunction with a snapshot of the pre-embedded vector store.

We use [elasticsearch-dump](https://github.com/elasticsearch-dump/elasticsearch-dump) to save and load bulk data from the vector store.
