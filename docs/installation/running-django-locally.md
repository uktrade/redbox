# Running Django locally

This guide can be followed once you have set up the application following steps in the [installation guide](../installation/index.md).

Hot reloading does not work for the Django app when run in Docker. This means every time you updated frontend code you will need to tear down and rebuild the container to see the changes reflected. To make frontend work easier you can run the Django app locally which allows hot reloading.

## **Env variables**

Add or update the environment variables MINIO_HOST, POSTGRES_HOST and UNSTRUCTURED_HOST to `localhost`

Set your COLLECTION_ENDPOINT to `"http://admin:Opensearch2024^@localhost:9200"`

## **Postgres**

Ensure you have a postgress database named `redbox-core` owned by a user called `redbox-core` as this is what the current POSTGRES_USER and POSTGRES_DB are looking for

## Running the server

Run any migrations
`poetry run python manage.py make migrations`
`poetry run python manage.py make migrate`

Run the server
`poetry run python manage.py runserver 8091`

## Frontend build

Install frontend dependencies if not done already `npm install`

Run `npm run dev` to watch for changes and automatically rebuild

### Quick start
Alternatively, run `make dev` to run migrations, copy static files and start the frontend with parcel in watch mode on [localhost:8081](http://localhost:8081/)
