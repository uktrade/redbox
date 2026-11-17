# 📮 Redbox

[![Python tests](https://github.com/uktrade/redbox/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/uktrade/redbox/actions/workflows/python-app.yml?query=branch%3Amain) [![Integration tests](https://github.com/uktrade/redbox/actions/workflows/integration.yml/badge.svg?branch=main)](https://github.com/uktrade/redbox/actions/workflows/integration.yml?query=branch%3Amain)

Redbox is a generative AI for working with documents, information and organisational knowledge.

Redbox is designed to help civil servants find, understand and use information more effectively. It uses large language models (LLMs) and retrieval-augmented generation (RAG) to facilitate information processing and research tasks.

Originally developed by the Incubator for AI (i.AI), Redbox is now developed and maintained by the Department for Business, Industry, Science and Trade (BIST).

## Service structure
The system has three main components:

|Component|Description|
|-|-|
|Django application|Handles the user interface, conversations, APIs, authentication and document management, among other things.|
|Worker|Worker for asynchronous document processing. Handles document ingestion, extraction, chunking, embedding and indexing.|
|Redbox AI engine|Handles LLM responsibilities such as LLM orchestration, retrieval, tools (capabilities provided to a LLM, such as web search or knowledge base retrieval), agents (collections of tools that can be used in combination to perform tasks), and response generation.|

These components interface with various external services for data storage, search, AI capabilities and other functionality.  All together, these constitute the Redbox service.


## Setup

Please refer to [developer_setup.md](./docs/developer_setup.md) for detailed instructions on setting up the project.

[The docs directory](./docs/) contains additional documentation about the service.

## Architecture & tech stack

Please refer to [architecture.md](./docs/architecture.md) for information about service architecture and tech stack.

## Contributing

We welcome contributions to this project. Please see [contributing.md](./docs/contributing.md) for more information.


## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
