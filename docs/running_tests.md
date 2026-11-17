
# Tests

## How to run tests

Tests are split into different commands based on the application the tests are for. For each application there is a separate `make` command to run those tests, these are:

For the django app when running in docker:

```bash
make test-django
```

For the django app when running locally:

```bash
make test-django-local
```

For the core AI:

```bash
make test-redbox
```

For integration tests:

```bash
make test-integration
```


## Notes on integration tests

There is a suite of playwright integration tests in the tests/ folder. There is a make command `test-integration` that will setup the local environment to be able to run the playwright tests, then execute the test suite.

The make command runs the entire suite in a headless configuration, however it is possible to run the playwright tests in debug mode using the playwright inspector. This command will start the playwright inspector tool: `make test-integration-debug`
