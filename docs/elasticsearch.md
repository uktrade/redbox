# Elasticsearch

## Installing Node and `elasticsearch-dump`

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

## Dumping data from Elasticsearch

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
