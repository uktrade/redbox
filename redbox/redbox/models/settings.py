import logging
import os
from functools import cache
from typing import Dict, Literal, Optional, Union
from urllib.parse import urlparse
from mcp.client.streamable_http import streamable_http_client
from mcp import ClientSession
from contextlib import asynccontextmanager

import boto3
from elasticsearch import Elasticsearch
from langchain.globals import set_debug
from opensearchpy import OpenSearch, Urllib3HttpConnection
from pydantic import AnyUrl, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from redbox_app.setting_enums import Environment

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger()

ENVIRONMENT = Environment[os.environ.get("ENVIRONMENT", "LOCAL").upper()]


class OpenSearchSettings(BaseModel):
    """settings required for a aws/opensearch"""

    model_config = SettingsConfigDict(frozen=True)

    collection_endpoint: str = os.environ.get("COLLECTION_ENDPOINT", "")
    parsed_url: AnyUrl = urlparse(collection_endpoint)

    logger.info(f"the parsed url is {parsed_url}")

    collection_endpoint__username: Optional[str] = parsed_url.username
    collection_endpoint__password: Optional[str] = parsed_url.password
    collection_endpoint__host: Optional[str] = parsed_url.hostname
    collection_endpoint__port: Optional[str] = "443"
    collection_endpoint__port_local: Optional[str] = "9200"  # locally, the port number is 9200


class ElasticLocalSettings(BaseModel):
    """settings required for a local/ec2 instance of elastic"""

    model_config = SettingsConfigDict(frozen=True)

    host: str = "elasticsearch"
    port: int = 9200
    scheme: str = "http"
    user: str = "elastic"
    version: str = "8.11.0"
    password: str = "redboxpass"
    subscription_level: str = "basic"


class ElasticCloudSettings(BaseModel):
    """settings required for elastic-cloud"""

    model_config = SettingsConfigDict(frozen=True)

    api_key: str
    cloud_id: str
    subscription_level: str = "basic"


class MCPServerSettings(BaseModel):
    model_config = SettingsConfigDict(frozen=True)
    name: str
    url: str
    secret_tokens: dict


class WebSearchSettings(BaseModel):
    model_config = SettingsConfigDict(frozen=True)
    name: str
    end_point: str
    secret_tokens: dict


class ChatLLMBackend(BaseModel):
    name: str = "anthropic.claude-3-7-sonnet-20250219-v1:0"
    provider: str = "bedrock"
    description: str | None = None
    model_config = {"frozen": True}


class Settings(BaseSettings):
    """Settings for the redbox application."""

    azure_api_version_embeddings: str = "2024-02-01"
    metadata_extraction_llm: ChatLLMBackend = ChatLLMBackend(
        name="anthropic.claude-3-sonnet-20240229-v1:0", provider="bedrock"
    )

    embedding_backend: str = os.environ.get("EMBEDDING_BACKEND", "amazon.titan-embed-text-v2:0")

    embedding_backend_vector_size: int = 1024

    embedding_max_retries: int = 1
    embedding_retry_min_seconds: int = 120  # Azure uses 60s
    embedding_retry_max_seconds: int = 300
    embedding_max_batch_size: int = 512
    embedding_document_field_name: str = "embedding"

    partition_strategy: Literal["auto", "fast", "ocr_only", "hi_res"] = "fast"
    clustering_strategy: Literal["full"] | None = None

    elastic: OpenSearchSettings = OpenSearchSettings()
    elastic_root_index: str = "redbox-data"
    elastic_chunk_alias: str = "redbox-data-chunk-current"

    kibana_system_password: str = "redboxpass"
    metricbeat_internal_password: str = "redboxpass"
    filebeat_internal_password: str = "redboxpass"
    heartbeat_internal_password: str = "redboxpass"
    monitoring_internal_password: str = "redboxpass"
    beats_system_password: str = "redboxpass"

    minio_host: str = "minio"
    minio_port: int = 9000
    aws_access_key: str | None = None
    aws_secret_key: str | None = None

    aws_region: str = "eu-west-2"
    bucket_name: str = "redbox-storage-dev"

    ## Chunks
    ### Normal
    worker_ingest_min_chunk_size: int = 1_000
    worker_ingest_max_chunk_size: int = 10_000
    ### Largest
    worker_ingest_largest_chunk_size: int = 300_000
    worker_ingest_largest_chunk_overlap: int = 0

    response_no_doc_available: str = "No available data for selected files. They may need to be removed and added again"
    response_max_content_exceeded: str = "Max content exceeded. Try smaller or fewer documents"

    object_store: str = "minio"

    dev_mode: bool = False
    superuser_email: str | None = None

    unstructured_host: str = "unstructured"

    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__", extra="allow", frozen=True)

    enable_metadata_extraction: bool = os.environ.get("ENABLE_METADATA_EXTRACTION")

    datahub_redbox_url: str = os.environ.get("DATAHUB_REDBOX_URL", "")
    datahub_redbox_secret_key: str = os.environ.get("DATAHUB_REDBOX_SECRET_KEY", "")
    datahub_redbox_access_key_id: str = os.environ.get("DATAHUB_REDBOX_ACCESS_KEY_ID", "")

    default_model_id: Optional[str] = os.environ.get("DEFAULT_MODEL_ID")

    allow_plan_feedback: bool = os.environ.get("ALLOW_PLAN_FEEDBACK", True)

    is_local: bool = ENVIRONMENT.is_local
    is_prod: bool = ENVIRONMENT.is_prod

    max_attempts: int = os.environ.get("MAX_ATTEMPTS", 3)

    # mcp
    caddy_mcp: MCPServerSettings = MCPServerSettings(
        name="caddy_mcp",
        url=os.environ.get("MCP_CADDY_URL", ""),
        secret_tokens={os.environ.get("MCP_HEADERS", ""): os.environ.get("MCP_CADDY_TOKEN", "")},
    )

    parlex_mcp: MCPServerSettings = MCPServerSettings(
        name="parlex_mcp",
        url=os.environ.get("MCP_PARLEX_URL", ""),
        secret_tokens={os.environ.get("MCP_HEADERS", ""): os.environ.get("MCP_PARLEX_TOKEN", "")},
    )

    datahub_mcp: MCPServerSettings = MCPServerSettings(
        name="datahub_mcp",
        url=os.environ.get("MCP_DATAHUB_URL", ""),
        secret_tokens={None: None},
    )
    # web search
    web_search: Literal["Google", "Brave", "Kagi"] = "Brave"

    ## Prompts
    metadata_prompt: tuple = (
        "system",
        "Given the first 1,000 tokens of a document and any available hard-coded file metadata, create"
        "SEO-optimized metadata for the document in the following JSON format:\n\n"
        '{{ "name": '
        ', "description": '
        ', "keywords": ["", "", "", "", ""] }}\n'
        "The description should summarize the document's content in a concise and SEO-friendly manner, "
        "and the keywords should represent the most relevant topics or phrases related to the document.",
        # "You are an SEO specialist that must optimise the metadata of a document "
        # "to make it as discoverable as possible. You are about to be given the first "
        # "1_000 tokens of a document and any hard-coded file metadata that can be "
        # "recovered from it. Create SEO-optimised metadata for this document."
        # "Description must be less than 100 words. and maximum 5 keywords .",
    )

    # Define index mapping for Opensearch - this is important so that KNN search works
    index_mapping: Dict = {
        "settings": {"index.knn": True},
        "mappings": {
            "properties": {
                "metadata": {
                    "properties": {
                        "chunk_resolution": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                        },
                        "created_datetime": {"type": "date"},
                        "creator_type": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                        },
                        "description": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                        },
                        "index": {"type": "long"},
                        "keywords": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                        },
                        "name": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                        },
                        "page_number": {"type": "long"},
                        "token_count": {"type": "long"},
                        "uri": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                        },
                        "uuid": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                        },
                    }
                },
                "text": {
                    "type": "text",
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                },
                "vector_field": {
                    "type": "knn_vector",
                    "dimension": embedding_backend_vector_size,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                    },
                },
            }
        },
    }

    @property
    def elastic_chat_mesage_index(self):
        return self.elastic_root_index + "-chat-mesage-log"

    @property
    def elastic_alias(self):
        return self.elastic_root_index + "-chunk-current"

    def get_agent_names(self):
        # get list of available agents
        from redbox.graph.agents.configs import agent_configs

        return [(agent.name, agent.name) for agent in agent_configs.values()]

    # @lru_cache(1) #removing cache because pydantic object (index mapping) is not hashable
    def elasticsearch_client(self) -> Union[Elasticsearch, OpenSearch]:
        logger.info("Testing OpenSearch is definitely being used")

        if ENVIRONMENT.is_local:
            client = OpenSearch(
                hosts=[
                    {
                        "host": self.elastic.collection_endpoint__host,
                        "port": self.elastic.collection_endpoint__port_local,
                    }
                ],
                http_auth=(
                    self.elastic.collection_endpoint__username,
                    self.elastic.collection_endpoint__password,
                ),
                use_ssl=False,
                connection_class=Urllib3HttpConnection,
            )

        else:
            client = OpenSearch(
                hosts=[
                    {
                        "host": self.elastic.collection_endpoint__host,
                        "port": self.elastic.collection_endpoint__port,
                    }
                ],
                http_auth=(
                    self.elastic.collection_endpoint__username,
                    self.elastic.collection_endpoint__password,
                ),
                use_ssl=True,
                verify_certs=True,
                connection_class=Urllib3HttpConnection,
                retry_on_timeout=True,
                pool_maxsize=100,
                timeout=120,
            )

        if not client.indices.exists_alias(name=self.elastic_alias):
            chunk_index = f"{self.elastic_root_index}-chunk"
            # client.options(ignore_status=[400]).indices.create(index=chunk_index)
            # client.indices.put_alias(index=chunk_index, name=self.elastic_alias)
            try:
                client.indices.create(
                    index=chunk_index, body=self.index_mapping, ignore=400
                )  # 400 is ignored to avoid index-already-exists errors
            except Exception as e:
                logger.error(f"Failed to create index {chunk_index}: {e}")

            try:
                client.indices.put_alias(index=chunk_index, name=f"{self.elastic_root_index}-chunk-current")
            except Exception as e:
                logger.error(f"Failed to set alias {self.elastic_root_index}-chunk-current: {e}")

        if not client.indices.exists(index=self.elastic_chat_mesage_index):
            try:
                client.indices.create(
                    index=self.elastic_chat_mesage_index, ignore=400
                )  # 400 is ignored to avoid index-already-exists errors
            except Exception as e:
                logger.error(f"Failed to create index {self.elastic_chat_mesage_index}: {e}")
            # client.indices.create(index=self.elastic_chat_mesage_index)

        return client

    @asynccontextmanager
    async def get_mcp_client(self, datahub_mcp_url: str):
        """
        Async context manager that yields a fully initialized MCP ClientSession.

        Usage:
            async with get_mcp_client("http://localhost:8100/mcp") as session:
                result = await session.call_tool("company_details", {"company_name": "BMW"})
        """
        async with streamable_http_client(datahub_mcp_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    def s3_client(self):
        if self.object_store == "minio":
            return boto3.client(
                "s3",
                aws_access_key_id=self.aws_access_key or "",
                aws_secret_access_key=self.aws_secret_key or "",
                endpoint_url=f"http://{self.minio_host}:{self.minio_port}",
            )

        if self.object_store == "s3":
            return boto3.client(
                "s3",
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.aws_region,
            )

        if self.object_store == "moto":
            from moto import mock_aws

            mock = mock_aws()
            mock.start()

            return boto3.client(
                "s3",
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.aws_region,
            )

        msg = f"unkown object_store={self.object_store}"
        raise NotImplementedError(msg)

    def web_search_settings(self):
        match self.web_search.lower():
            case "google":
                return WebSearchSettings(
                    name="Google",
                    end_point="https://customsearch.googleapis.com/customsearch/v1",
                    secret_tokens={
                        "key": os.environ.get("GOOGLE_SEARCH_API", ""),
                        "cx": os.environ.get("GOOGLE_SEARCH_ENGINE", ""),
                    },
                )
            case "brave":
                return WebSearchSettings(
                    name="Brave",
                    end_point="https://api.search.brave.com/res/v1/web/search",
                    secret_tokens={
                        "Accept": "application/json",
                        "Accept-Encoding": "gzip",
                        "x-subscription-token": os.environ.get("BRAVE_API_KEY", ""),
                    },
                )
            case "kagi":
                return WebSearchSettings(
                    name="Kagi",
                    end_point="https://kagi.com/api/v0/search",
                    secret_tokens={"Authorization": " ".join(["Bot", os.environ.get("KAGI_API_KEY", "")])},
                )


@cache
def get_settings() -> Settings:
    s = Settings()
    set_debug(s.dev_mode)
    return s
