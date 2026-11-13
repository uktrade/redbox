# Architecture <!-- omit in toc -->

## Disclaimer <!-- omit in toc -->

This document describes the architecture of a continuously changing service. As such, aspects of this document will probably be out of date at time of reading.
- Use this document as a guide to the service architecture - what it does, why, and roughly how.
- Examine the codebase to understand current implementation details.

## Contents <!-- omit in toc -->


- [Principles](#principles)
- [How it works at a glance](#how-it-works-at-a-glance)
- [Components](#components)
  - [Django app](#django-app)
  - [Worker](#worker)
    - [Scaling](#scaling)
    - [Django-Q2](#django-q2)
    - [Workflow](#workflow)
  - [AI engine](#ai-engine)
  - [Research multi-agent graph (@newroute)](#research-multi-agent-graph-newroute)
  - [MCP (Model Context Protocol)](#mcp-model-context-protocol)
- [Document ingestion](#document-ingestion)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
  - [Key files for understanding the codebase](#key-files-for-understanding-the-codebase)


## Principles

Redbox is designed to follow these core architectural principles where possible:

- **Modularity**: Components and services can be swapped out independently.
- **Cloud Agnostic**: The current deployment targets AWS, but each component can be replaced independently to suit a different cloud provider or an on-premises setup.
- **Scalability**: The system scales horizontally — additional instances can be added to handle increased load.
- **Resilience**: The system recovers from individual instance failures without losing availability.
- **Security**: Unauthorised access is prevented at all layers.
- **Simplicity**: A simple approach to data ownership lets deployers introduce their own business logic separately from the core system.

## How it works at a glance

Redbox is a [retrieval-augmented generation (RAG)](https://www.anthropic.com/engineering/contextual-retrieval) application: instead of relying only on what the LLM already knows, it looks up relevant passages from the user's uploaded files (and other sources) and feeds them into the prompt alongside the question. To make that lookup fast and meaning-aware, each document is split into smaller **chunks** and converted into **embeddings** — numeric vectors where chunks with similar meaning end up close together in vector space. At query time, the user's question is embedded the same way, and the closest chunks are retrieved and passed to the LLM as context. This is why the architecture has both an **ingestion path** (extract → chunk → embed → index) and a **chat path** (embed query → retrieve chunks → generate answer).

## Components

Redbox has three main components:

- **Django app** (`django_app/`) — a web application that manages authentication, real-time chat, and file uploads — storing files in S3 and queuing ingestion tasks in PostgreSQL
- **Worker (document ingestion)** (`django_app/redbox_app/worker.py`) — a background process that extracts and chunks the file uploads, generates embeddings using [Amazon Titan Embed Text v2](https://aws.amazon.com/bedrock/titan/) via [AWS Bedrock](https://aws.amazon.com/bedrock/), and indexes them into [OpenSearch](https://opensearch.org/)
- **AI engine** (`redbox/`) — a [LangGraph](https://langchain-ai.github.io/langgraph/) state machine that generates responses using [Claude 3.7 Sonnet](https://www.anthropic.com/claude) via AWS Bedrock and retrieves relevant document chunks from OpenSearch


### Django app

The `django_app/` directory contains both the server-side application and the browser-side chat UI. The frontend (`django_app/frontend/`) is a set of [Web Components](https://developer.mozilla.org/en-US/docs/Web/API/Web_components) written in JavaScript and bundled by [Parcel](https://parceljs.org/). [WhiteNoise](https://whitenoise.readthedocs.io/) serves the compiled assets as static files, which execute client-side in the browser.

The server (`django_app/redbox_app/`) runs behind [Daphne](https://github.com/django/daphne), an [ASGI](https://asgi.readthedocs.io/) server. ASGI is the async successor to WSGI and supports long-lived connections like [WebSockets](https://en.wikipedia.org/wiki/WebSocket) alongside regular HTTP. WebSockets are used here to stream LLM responses token-by-token to the browser in real time,  without waiting for the entire answer to be generated.

Each incoming chat message is handled by [`ChatConsumer`](../../django_app/redbox_app/redbox_core/consumers.py), a [Django Channels](https://channels.readthedocs.io/) `AsyncWebsocketConsumer`. Channels is Django's extension for WebSockets and other long-lived async connections. It calls the AI engine's LangGraph state machine, persists chat sessions, messages, citations, and agent plans to PostgreSQL, and pushes tokens and events back to the browser in real time.

Authentication is handled by [`django-staff-sso-client`](https://github.com/uktrade/django-staff-sso-client) (BIST Staff SSO via OAuth). A custom `TokenCaptureBackend` extends the SSO backend to capture the OAuth token into the session for downstream use.

File uploads follow a different path from chat messages. Ingestion can take minutes, so it runs out-of-band: a Django [view](http://docs.djangoproject.com/en/6.0/topics/http/views/) (an HTTP request handler) stores the file in S3, writes a *task* row to PostgreSQL describing the work, and returns immediately. A separate **worker** process picks the task up and runs it — see [Document ingestion](#document-ingestion). This producer/worker pattern is implemented by [Django-Q2](https://django-q2.readthedocs.io/), which uses PostgreSQL as the task store and so avoids the need for a separate message broker (e.g. Redis or RabbitMQ).

File uploads can optionally be scoped to a [`Tool`](../../django_app/redbox_app/redbox_core/models.py) — a named workspace that bundles a knowledge base (uploaded files), team/user access controls, and a set of AI agents under a shared URL (`/tools/<slug>/`). Different teams can have separate Tools for different purposes (e.g. "Ministerial Submissions", "Legislation Search").

### Worker

The Worker is a set of service that runs in the background and perform tasks that are too resource-intensive to run in the Django application. These tasks include:

- Ingesting files
- Embedding chunks of text

The Worker is designed to be horizontally scalable. This means that we can add more instances of the Worker to handle more load. The Worker is also designed to be resilient. If one instance of the Worker fails, another instance can process queued tasks (dependant on the queue's retry and failure-handling configuration).

Tasks are queued in the PostgreSQL database.

#### Scaling

When running locally there is only one redbox-worker container running a Django-Q2 cluster. By default the cluster runs five workers; This can be adjusted by setting environment variable Q_WORKERS.

When running in AWS we can scale the number of ECS worker tasks up or down depending on the load. Tasks are queued in PostgreSQL.

#### Django-Q2

We use [Django_q2](https://faststream.airt.ai/latest/faststream/) to manage background tasks. It uses PostgreSQL for its task queue, avoiding the need for a separate message broker.The Django application adds tasks to the queue, then the Worker runs the qcluster process to pick up and execute those tasks.

#### Workflow

The Worker is responsible for file ingestion and embedding. These are both part of a single pipeline that executes when a file is first uploaded.

1. **Ingesting files into the system.** When a file is uploaded, the Django application stores it in object storage (s3), creates a record for the file in PostgreSQL, and queues an ingestion task in postgreSQL. When the Worker picks up that task, it retrieves the file from s3, performs any required format conversion, then extracts and chunks the text.
2. **Embedding chunks of text.** As part of the ingestion process the worker generates vector embeddings for the extracted text using an embedding model, such as Amazon Titan, through AWS Bedrock. The chunks and their embeddings are indexed in OpenSearch for later retrieval.
3. **Managing file status in the database.** When ingestion finishes the worker updates the file's `status` property in PostgreSQL to indicate whether the process completed successfully or failed.


### AI engine

The `redbox/` directory defines the AI engine: a [LangGraph](https://langchain-ai.github.io/langgraph/) state machine implemented using [`StateGraph`](https://langchain-ai.github.io/langgraph/concepts/low_level/#stategraph). A LangGraph state machine is a directed graph of nodes that each read and update a shared state object, with edges deciding which node runs next — a structure that lets the engine orchestrate LLM calls, retrieval, and agent coordination in a single flow. It receives a chat message from `ChatConsumer` and produces a streamed response.

The state machine routes the message to one of several subgraphs. Following the [LangGraph workflow/agent distinction](https://docs.langchain.com/oss/python/langgraph/workflows-agents), these fall into two categories: **workflows**, where the execution path is predetermined, and **agents**, where an LLM dynamically decides what to do next. The default route is `newroute`, a [multi-agent graph](https://docs.langchain.com/oss/python/langchain/multi-agent) used for research described [below](#research-multi-agent-graph). Users can instead prefix their message with `@<route>` to invoke one of the workflows below, which are faster and cheaper and give explicit control over retrieval strategy.

| Workflow | Behaviour |
| --- | --- |
| `@search` | Embeds the query, retrieves the most relevant chunks from the user's selected files, and answers with structured citations |
| `@summarise` | Retrieves large (up to 300K token) full-document chunks and applies map-reduce summarisation |
| `@chat` | Plain LLM chat with no document retrieval |

### Research multi-agent graph (@newroute)

The research graph uses a [router pattern](https://docs.langchain.com/oss/python/langchain/multi-agent/router) with multiple [agents](https://docs.langchain.com/oss/python/langchain/agents) to support more complex queries. A Planner breaks the query into tasks, Workers gather information via [agent tools](https://docs.langchain.com/oss/python/langchain/tools) — searching documents, querying data, or fetching from external APIs — and an Evaluator synthesises the results, with raw source documents returned as citations.

At time of writing, these are some of the available agents:

| Agent | Role | Purpose | Agent tools |
| --- | --- | --- | --- |
| `Planner_Agent` | Planner | Generates a task plan for the user to approve or modify | — |
| `Internal_Retrieval_Agent` | Worker | KNN search over user-uploaded files | `search_documents` |
| `Knowledge_Base_Retrieval_Agent` | Worker | Search and SQL query over Tool knowledge base files (tabular and non-tabular) | `search_knowledge_base`, `query_tabular_knowledge_base_file` |
| `Tabular_Agent` | Worker | SQL queries against user-uploaded CSV and Excel files | `query_tabular_file` |
| `Summarisation_Agent` | Worker | Full-document summarisation via a retrieval subgraph; streams result directly to the user | — |
| `External_Retrieval_Agent` | Worker | Wikipedia and gov.uk search | `search_wikipedia`, `search_govuk` |
| `Web_Search_Agent` | Worker | General web search | `web_search` |
| `Legislation_Search_Agent` | Worker | legislation.gov.uk search | `legislation_search` |
| `Artifact_Builder_Agent` | Worker | Retrieve artifact templates (files prefixed `Artifact_`) from the knowledge base | `retrieve_specific_files_knowledge_base` |
| `Submission_Checker_Agent` | Worker | Evaluate and check ministerial submissions; provides structured quality assessment | `retrieve_document_full_text`, `retrieve_knowledge_base`, `retrieve_document_from_prompt` |
| `Submission_Question_Answer_Agent` | Worker | Answer questions about submissions and follow-up questions on prior evaluations | `retrieve_document_full_text`, `retrieve_knowledge_base`, `retrieve_document_from_prompt` |
| `Datahub_Agent` | Worker | Query company data from Datahub via MCP (see below) | Datahub MCP tools |
| `Evaluator_Agent` | Evaluator | Synthesises worker results into a final response | — |

### MCP (Model Context Protocol)

Worker agents can use [MCP](https://modelcontextprotocol.io/) to call external services. Redbox integrates MCP via [`langchain-mcp-adapters`](https://github.com/langchain-ai/langchain-mcp-adapters) to expose those services as agent tools. Three MCP servers are configured in `redbox/models/settings.py`; only `datahub_mcp` is currently wired to an agent.

The `Datahub_Agent` runs a tool-calling loop up to a set number of iterations: the LLM selects tools, executes them in parallel, formats each result, and feeds it back until it has enough information to produce a final answer. Each tool execution opens a fresh MCP session authenticated with the user's SSO Bearer token. If a tool call requires clarification, the agent pauses and surfaces the question directly to the user. The final response is parsed and formatted as documents with deep links — URLs that point directly to the relevant record or page within Datahub.


## Document ingestion

Document ingestion is the **ingestion path** introduced earlier — the offline pipeline that prepares an uploaded file so the AI engine can later retrieve relevant passages from it. It is triggered by the task that the Django view enqueued on upload, and runs in the Django-Q2 worker process. The worker pulls the next task from PostgreSQL, looks up the file in S3, extracts its text using a format-specific tool, uses an LLM to generate file-level metadata, splits the text into chunks, embeds those chunks, and indexes both the chunks and their vectors in OpenSearch. When the pipeline finishes it writes the final status — `complete` or `errored` — back to PostgreSQL, which the UI polls to show progress.

```mermaid
flowchart LR
    subgraph Stores[" "]
        S3[\"AWS S3"/]
        PG[("PostgreSQL")]
    end

    S3 -- read file --> MetaLLM & FormatCheck
    PG -- file_id & file record --> FormatCheck

    subgraph OpenSearch["OpenSearch"]
        ES[("chunk index")]
        ESSchema[("schematised <br>index")]
    end
    subgraph Worker["Document ingestion (worker.py)"]
        MetaLLM["MetadataLoader"]
        FormatCheck{"Format<br>detection"}
        Unstructured["unstructured"]
        LibreOffice["LibreOffice<br>.doc → .docx"]
        Schema["pandas"]
        Chunk["1K–10K character chunks"]
        Textract([AWS Textract])
    end
    subgraph "AWS Bedrock"
        Claude([Claude])
        Titan([Amazon Titan])
    end

    FormatCheck -- ".doc" --> LibreOffice
    LibreOffice -- ".docx" --> Unstructured
    FormatCheck -- DOCX · PPT · PPTX · TXT --> Unstructured
    FormatCheck -- PDF --> Textract
    FormatCheck -- CSV · TSV · XLSX · XLS --> Schema
    Textract -- extracted text --> Chunk
    Unstructured -- extracted text --> Chunk
    MetaLLM <-- metadata inference --> Claude
    MetaLLM -- metadata --> Chunk
    Chunk -- "embed" --> Titan
    Titan -- "vectors" --> ES
    Chunk -- "chunks (no embed)" --> ES
    Schema --"row text"--> Chunk
    Schema -- schema + samples --> ESSchema
```

**Text extraction**: PDF text is extracted using a configurable extraction process. **Note that this process is subject to change. he configuration documented here is true at time of writing, but is subject to change, so consider it as an example rather than the actual process**.  
At time of writing, [AWS Textract](https://aws.amazon.com/textract/), is attempted first, followed by Unstructured extraction, then PyMuPDF is a final fallback. LArge PDFs are handled separately using AWS Textract, with PyMuPDF as a fallback. DOCX and PPTX files are converted to PDF using LibreOffice, then they pass through the same pipeline. Text files are handled by Unstructured. Tabular files are parsed by pandas.

**Metadata extraction**: `MetadataExtraction` uses the configured metadata extraction LLM to process up to the first 10K characters of each file to generate a name, description, and keywords. For tabular files schema information and sample data are also processed. This metadata is embedded into every chunk and used by agents to understand what files are available.

**Indexing**: Chunks are indexed in OpenSearch in two ways:

- **With embeddings** — 1024-dimension vectors generated by [Amazon Titan Embed Text v2](https://aws.amazon.com/bedrock/titan/) via AWS Bedrock, for [KNN semantic search](https://opensearch.org/docs/latest/search-plugins/knn/index/) used by `@search` and the research agents
- **Without embeddings** — for full-text retrieval during whole-document summarisation via `@summarise`

**Tabular files**: CSV, TSV, and Excel files are additionally parsed by pandas to extract column schemas and row samples, stored in a separate schematised OpenSearch index (`redbox-schematised`). At query time, the `Tabular_Agent` and `Knowledge_Base_Retrieval_Agent` (see [AI engine](#ai-engine)) reconstruct a DuckDB database from these schemas and execute agent-generated SQL against it.

## Tech stack

| Layer | Technology |
| --- | --- |
| LLM | [AWS Bedrock](https://aws.amazon.com/bedrock/) — [Claude 3.7 Sonnet](https://www.anthropic.com/claude) (primary), Claude 3 Sonnet (metadata extraction) |
| Embeddings | [Amazon Titan Embed Text v2](https://aws.amazon.com/bedrock/titan/) (1024-dim) |
| AI orchestration | [LangGraph](https://langchain-ai.github.io/langgraph/) + [LangChain](https://python.langchain.com/) + [MCP](https://modelcontextprotocol.io/) (`langchain-mcp-adapters`) |
| Vector store | [OpenSearch](https://opensearch.org/) (KNN search with Gaussian score boosting; separate schematised index for tabular column schemas) |
| Tabular query engine | [DuckDB](https://duckdb.org/) (in-process SQL queries over CSV and Excel data) |
| Document processing | [AWS Textract](https://aws.amazon.com/textract/) (PDF extraction), [unstructured](https://github.com/Unstructured-IO/unstructured) (DOCX, PPTX, TXT extraction), [LibreOffice](https://www.libreoffice.org/) (`.doc` → `.docx`) |
| Backend framework | [Django 5](https://www.djangoproject.com/) + [Django Channels](https://channels.readthedocs.io/) |
| ASGI server | [Daphne](https://github.com/django/daphne) + [WhiteNoise](https://whitenoise.readthedocs.io/) (static files) |
| Authentication | `django-staff-sso-client` (BIST Staff SSO, primary)
| Email | [GOV.UK Notify](https://www.notifications.service.gov.uk/) (`django-gov-notify`) |
| Feature flags | [Django-Waffle](https://waffle.readthedocs.io/) |
| Admin dashboards | [Plotly Dash](https://dash.plotly.com/) (`django-plotly-dash`) |
| File storage | [AWS S3](https://aws.amazon.com/s3/) |
| Relational database | [PostgreSQL](https://www.postgresql.org/) |
| Task queue | [Django-Q2](https://django-q2.readthedocs.io/) |
| Frontend | Web Components ([Parcel](https://parceljs.org/)-bundled), JavaScript, Sass (`django-libsass`), [GOV.UK Frontend](https://frontend.design-system.service.gov.uk/) 5.10, [Playwright](https://playwright.dev/) (E2E tests) |
| Observability | [Sentry](https://sentry.io/), [Datadog](https://www.datadoghq.com/) (`ddtrace`), [OpenTelemetry](https://opentelemetry.io/) |
| Infrastructure | [Docker](https://www.docker.com/), [AWS ECS](https://aws.amazon.com/ecs/) + [Terraform](https://www.terraform.io/) |

---

## Project structure

```txt
redbox/
├── django_app/                      # Django web application
│   ├── frontend/                    # Parcel-bundled web components
│   │   ├── src/                     # Web component source (JS/CSS)
│   │   └── tests-web-components/    # Web component tests
│   ├── redbox_app/                  # Django application package
│   │   ├── redbox_core/             # Core Django app
│   │   │   ├── consumers.py         # WebSocket handler (chat entry point)
│   │   │   ├── models.py            # ORM models (File, Chat, Citation, …)
│   │   │   ├── views/               # HTTP views
│   │   │   ├── migrations/          # Database migrations
│   │   │   └── management/          # Management commands
│   │   ├── settings.py
│   │   ├── urls.py
│   │   └── worker.py                # Ingestion worker
│   ├── static/                      # Compiled static assets
│   ├── tests/                       # Django app test suite
│   ├── manage.py
│   ├── pyproject.toml
│   └── Dockerfile
├── redbox/                          # AI engine
│   ├── redbox/                      # Python package
│   │   ├── graph/                   # LangGraph state machines and agents
│   │   │   ├── root.py              # Graph definition — nodes, edges, routing
│   │   │   ├── agents/              # Agent implementations (workers, planner, …)
│   │   │   └── nodes/               # Individual graph node functions
│   │   ├── api/                     # API wrappers and streaming callbacks
│   │   ├── chains/                  # LLM chains and ingestion logic
│   │   ├── loader/                  # Document loaders and preprocessing
│   │   ├── models/                  # Pydantic data models and settings
│   │   ├── retriever/               # OpenSearch retrieval and query builders
│   │   ├── test/                    # Shared test fixtures and helpers
│   │   ├── app.py                   # Redbox class — main entry point
│   │   └── transform.py
│   ├── tests/                       # Library test suite
│   └── pyproject.toml
├── tests/                           # End-to-end / integration tests
├── docs/                            # Project documentation
├── notebooks/                       # Jupyter notebooks
├── utilities/                       # Developer utilities (e.g. draw_graph.py)
├── docker-compose.yml
├── pyproject.toml
├── Makefile
└── README.md
```

### Key files for understanding the codebase

| File | Purpose |
| --- | --- |
| [`django_app/redbox_app/redbox_core/consumers.py`](../../django_app/redbox_app/redbox_core/consumers.py) | WebSocket handler — entry point for all chat messages |
| [`redbox/redbox/app.py`](../../redbox/redbox/app.py) | `Redbox` class — wires up the LangGraph graph, retrievers, and agent tools |
| [`redbox/redbox/graph/root.py`](../../redbox/redbox/graph/root.py) | Graph definition — nodes, edges, and route branching |
| [`redbox/redbox/graph/agents/workers.py`](../../redbox/redbox/graph/agents/workers.py) | Worker agent execution — how a task is run with tool calls |
| [`redbox/redbox/retriever/retrievers.py`](../../redbox/redbox/retriever/retrievers.py) | OpenSearch retriever — KNN search with filters and Gaussian boosting |
| [`django_app/redbox_app/worker.py`](../../django_app/redbox_app/worker.py) | Ingestion worker — preprocessing, chunking, embedding, and indexing |
