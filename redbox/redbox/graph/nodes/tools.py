import json
import logging
import random
import sqlite3
import time
from typing import Annotated, Callable, Iterable, Union

import boto3
import numpy as np
import requests
from elasticsearch import Elasticsearch
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.documents import Document
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.messages import ToolCall
from langchain_core.tools import Tool, tool
from langgraph.prebuilt import InjectedState
from mohawk import Sender
from opensearchpy import OpenSearch
from sklearn.metrics.pairwise import cosine_similarity
from waffle.decorators import waffle_flag

from redbox.api.format import format_documents
from redbox.chains.components import get_embeddings
from redbox.models.chain import RedboxState
from redbox.models.file import ChunkCreatorType, ChunkMetadata, ChunkResolution
from redbox.models.settings import get_settings
from redbox.retriever.queries import (
    add_document_filter_scores_to_query,
    build_document_query,
    get_all,
    get_knowledge_base,
)
from redbox.retriever.retrievers import query_to_documents
from redbox.transform import bedrock_tokeniser, merge_documents, sort_documents

log = logging.getLogger(__name__)


def format_result(loop, content, artifact, status, is_intermediate_step):
    if loop:
        return ((content, status, str(is_intermediate_step)), artifact)
    else:
        return (content, artifact)


def build_document_from_prompt_tool(loop: bool = False):
    @tool(response_format="content_and_artifact")
    def _retrieve_document_from_prompt(
        state: Annotated[RedboxState, InjectedState], is_intermediate_step: bool = False
    ) -> tuple:
        """
        Retrieve document from user prompt

        Arg:
        - is_intermediate_step (bool): True if this tool call is an intermediate step to allow you to gather information from user prompt. False if this is your final step.

        Return:
            Tuple: document
        """
        return format_result(
            loop=loop,
            content="<context>This is user prompt that containing documents.</context>" + state.request.question,
            artifact=[],
            status="pass",
            is_intermediate_step=is_intermediate_step,
        )

    return _retrieve_document_from_prompt


def build_retrieve_document_full_text(es_client: Union[Elasticsearch, OpenSearch], index_name: str, loop: bool = False):
    @tool(response_format="content_and_artifact")
    def _retrieve_document_full_text(
        state: Annotated[RedboxState, InjectedState], is_intermediate_step: bool = False
    ) -> tuple:
        """
        Retrieve full texts from state.documents. This tool should be used when a full text from a document is required.
        This tool does not retrieve documents in knowledge base.

        Arg:
        - is_intermediate_step (bool): True if this tool call is an intermediate step to allow you to gather information about the document. False if this is your final step.

        Return:
            Tuple: Collection of matching document full texts with metadata
        """

        el_query = get_all(chunk_resolution=ChunkResolution.largest, state=state)

        results = query_to_documents(es_client=es_client, index_name=index_name, query=el_query)
        if not results:
            return format_result(
                loop=loop,
                content="Tool returns empty result set.",
                artifact=[],
                status="fail",
                is_intermediate_step=is_intermediate_step,
            )

        # Return as state update
        sorted_documents = sorted(results, key=lambda result: result.metadata["index"])
        return format_result(
            loop=loop,
            content="<context>This is user full text documents.</context>" + format_documents(sorted_documents),
            artifact=sorted_documents,
            status="pass",
            is_intermediate_step=is_intermediate_step,
        )

    return _retrieve_document_full_text


def build_retrieve_knowledge_base(es_client: Union[Elasticsearch, OpenSearch], index_name: str, loop: bool = False):
    @tool(response_format="content_and_artifact")
    def _retrieve_knowledge_base(
        state: Annotated[RedboxState, InjectedState], is_intermediate_step: bool = False
    ) -> tuple[str, list[Document]]:
        """
        Retrieve knowledge base data, information for this agent.

        Arg:
        - is_intermediate_step (bool): True if this tool call is an intermediate step to allow you to gather knowledge base. False if this is your final step.

        Return:
            Tuple: Collection of knowledge base documents with metadata
        """
        el_query = get_knowledge_base(chunk_resolution=ChunkResolution.largest, state=state)
        results = query_to_documents(es_client=es_client, index_name=index_name, query=el_query)

        if not results:
            return format_result(
                loop=loop,
                content="Tool returns empty result set.",
                artifact=[],
                status="fail",
                is_intermediate_step=is_intermediate_step,
            )

        # Return as state update
        sorted_documents = sorted(results, key=lambda result: result.metadata["index"])
        return format_result(
            loop=loop,
            content="<context>This is your knowledgebase result.</context>" + format_documents(sorted_documents),
            artifact=sorted_documents,
            status="pass",
            is_intermediate_step=is_intermediate_step,
        )

    return _retrieve_knowledge_base


def build_search_documents_tool(
    es_client: Union[Elasticsearch, OpenSearch],
    index_name: str,
    embedding_model: Embeddings,
    embedding_field_name: str,
    chunk_resolution: ChunkResolution | None,
) -> Tool:
    """Constructs a tool that searches the index and sets state.documents."""

    @tool(response_format="content_and_artifact")
    def _search_documents(query: str, state: Annotated[RedboxState, InjectedState]) -> tuple[str, list[Document]]:
        """
        "Searches through state.documents to find and extract relevant information. This tool should be used whenever a query involves finding, searching, or retrieving information from documents that have already been uploaded or provided to the system.

        The tool performs semantic search across all available documents. Results are automatically grouped by source document and ranked by relevance score. Each result includes document metadata (title, page/section) for context.

        Args:
            query (str): The search query to match against document content.
            - Can be natural language, keywords, or phrases
            - More specific queries yield more precise results
            - Query length should be 1-500 characters
        Returns:
            dict[str, Any]: Collection of matching document snippets with metadata:
        """
        start_time = time.time()
        query_vector = embedding_model.embed_query(query)
        selected_files = state.request.s3_keys
        permitted_files = state.request.permitted_s3_keys
        ai_settings = state.request.ai_settings

        # Initial pass
        initial_query = build_document_query(
            query=query,
            query_vector=query_vector,
            selected_files=selected_files,
            permitted_files=permitted_files,
            embedding_field_name=embedding_field_name,
            chunk_resolution=chunk_resolution,
            ai_settings=ai_settings,
        )
        initial_documents = query_to_documents(es_client=es_client, index_name=index_name, query=initial_query)
        log.debug("Initial query using %s seconds", time.time() - start_time)

        # Handle nothing found (as when no files are permitted)
        if not initial_documents:
            return "", []

        # Adjacent documents
        with_adjacent_query = add_document_filter_scores_to_query(
            elasticsearch_query=initial_query,
            ai_settings=ai_settings,
            centres=initial_documents,
        )
        adjacent_boosted = query_to_documents(es_client=es_client, index_name=index_name, query=with_adjacent_query)
        log.debug("Adjacent boosted query using %s seconds", time.time() - start_time)

        # Merge and sort
        merged_documents = merge_documents(initial=initial_documents, adjacent=adjacent_boosted)
        sorted_documents = sort_documents(documents=merged_documents)
        log.debug("Merge and sort documents using %s seconds", time.time() - start_time)

        # Return as state update
        return format_documents(sorted_documents), sorted_documents

    return _search_documents


def build_govuk_search_tool(filter=True) -> Tool:
    """Constructs a tool that searches gov.uk and sets state["documents"]."""

    tokeniser = bedrock_tokeniser

    def recalculate_similarity(response, query, num_results):
        embedding_model = get_embeddings(get_settings())
        em_query = embedding_model.embed_query(query)
        for r in response.get("results"):
            text_compare = r.get("description") if r.get("description") else r.get("indexable_content")[:500]
            em_des = embedding_model.embed_query(text_compare)
            r["similarity"] = cosine_similarity(np.array(em_query).reshape(1, -1), np.array(em_des).reshape(1, -1))[0][
                0
            ]
        response["results"] = sorted(response.get("results"), key=lambda x: x["similarity"], reverse=True)[:num_results]
        return response

    @tool(response_format="content_and_artifact")
    def _search_govuk(query: str, state: Annotated[RedboxState, InjectedState]) -> tuple[str, list[Document]]:
        """
        Search for documents on www.gov.uk based on a query string.
        This endpoint is used to search for documents on www.gov.uk. There are many types of documents on www.gov.uk.
        Types include:
        - guidance
        - policy
        - legislation
        - news
        - travel advice
        - departmental reports
        - statistics
        - consultations
        - appeals

        Args:
            query (str): The query for searching on GOV.UK web site.
            - Can be natural language, keywords, or phrases
            - More specific queries yield more precise results
            - Query length should be 1-500 characters
        Returns:
            dict[str, Any]: Collection of matching document snippets with metadata:

        """
        if len(query) > 0:
            max_content_tokens = 1000
            url_base = "https://www.gov.uk"
            required_fields = [
                "format",
                "title",
                "description",
                "indexable_content",
                "link",
            ]
            ai_settings = state.request.ai_settings
            response = requests.get(
                f"{url_base}/api/search.json",
                params={
                    "q": query,
                    "count": (
                        ai_settings.tool_govuk_retrieved_results if filter else ai_settings.tool_govuk_returned_results
                    ),
                    "fields": required_fields,
                },
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            response = response.json()

            if filter:
                response = recalculate_similarity(response, query, ai_settings.tool_govuk_returned_results)

            mapped_documents = []
            for i, doc in enumerate(response["results"]):
                if any(field not in doc for field in required_fields):
                    continue
                # truncate content
                content = doc["indexable_content"][:max_content_tokens]
                mapped_documents.append(
                    Document(
                        page_content=content,
                        metadata=ChunkMetadata(
                            index=i,
                            uri=f"{url_base}{doc['link']}",
                            token_count=tokeniser(content),
                            creator_type=ChunkCreatorType.gov_uk,
                        ).model_dump(),
                    )
                )

            return format_documents(mapped_documents), mapped_documents
        else:
            # no query for search
            return "", []

    return _search_govuk


def build_search_wikipedia_tool(number_wikipedia_results=1, max_chars_per_wiki_page=12000) -> Tool:
    """Constructs a tool that searches Wikipedia"""
    _wikipedia_wrapper = WikipediaAPIWrapper(
        top_k_results=number_wikipedia_results,
        doc_content_chars_max=max_chars_per_wiki_page,
    )
    tokeniser = bedrock_tokeniser

    @tool(response_format="content_and_artifact")
    def _search_wikipedia(query: str) -> tuple[str, list[Document]]:
        """
        Search Wikipedia for information about the queried entity.
        Useful for when you need to answer general questions about people, places, objects, companies, facts, historical events, or other subjects.
        Input should be a search query.

        Args:
            query (str): The search query string used to find pages.
                This could be a keyword, phrase, or name

        Returns:
            response (str): The content of the relevant Wikipedia page
        """
        response = _wikipedia_wrapper.load(query)
        if not response:
            print("No Wikipedia response found.")
            return "", []

        mapped_documents = []
        for i, doc in enumerate(response):
            token_count = tokeniser(doc.page_content)
            print(f"Document {i} token count: {token_count}")

            mapped_documents.append(
                Document(
                    page_content=doc.page_content,
                    metadata=ChunkMetadata(
                        index=i,
                        uri=doc.metadata["source"],
                        token_count=token_count,
                        creator_type=ChunkCreatorType.wikipedia,
                    ).model_dump(),
                )
            )
        docs = mapped_documents
        return format_documents(docs), docs

    return _search_wikipedia


@waffle_flag("DATA_HUB_API_ROUTE_ON")
def parse_filters_bedrock(prompt: str):
    client = boto3.client("bedrock-runtime", region_name="eu-west-2")

    settings = get_settings()

    model_id = settings.default_model_id if settings.default_model_id else "anthropic.claude-3-sonnet-20240229-v1:0"

    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"You are a data filter generator for Data Hub.\n"
                            f'Based on this question: "{prompt}", respond with a JSON object containing:\n'
                            f" - dataset: one of [companies-dataset, contacts-dataset, events-dataset, "
                            f"interactions-dataset, investment-projects-dataset]\n"
                            f" - filters: a dictionary of relevant filters. These include:\n"
                            f"   - For companies-dataset: address_1, address_2, address_county, address_country__name, address_postcode, address_area__name, address_town, archived, archived_on, archived_reason, business_type__name, company_number, created_by_id, created_on, description, duns_number, export_experience_category__name, global_headquarters_id, global_ultimate_duns_number, headquarter_type__name, id, is_number_of_employees_estimated, is_turnover_estimated, modified_on, name, number_of_employees, one_list_account_owner_id, one_list_tier__name, reference_code, registered_address_1, registered_address_2, registered_address_country__name, registered_address_county, registered_address_postcode, registered_address_area__name, registered_address_town, export_segment, export_sub_segment, trading_names, turnover, uk_region__name, vat_number, website, is_out_of_business, strategy, sector_name, Consumer and retail, one_list_core_team_advisers, turnover_gbp, etc.\n"
                            f"   - For contacts-dataset: address_1, address_2, address_country__name, address_county, address_postcode, address_same_as_company, address_town, archived, archived_on, company_id, created_by_id, created_on, email, first_name, id, job_title, last_name, modified_on, notes, primary, full_telephone_number, valid_email, name, etc.\n"
                            f"   - For events-dataset: address_1, address_2, address_country__name, address_county, address_postcode, address_town, created_by_id, created_on, disabled_on, end_date, event_type__name, id, lead_team_id, location_type__name, name, notes, organiser_id, start_date, uk_region__name, service_name, team_ids, related_programme_names, etc.\n"
                            f"   - For interactions-dataset: communication_channel__name, company_id, created_by_id, created_on, date, event_id, grant_amount_offered, id, investment_project_id, company_export_id, kind, modified_on, net_company_receipt, notes, policy_feedback_notes, service_delivery_status__name, subject, theme, were_countries_discussed, export_barrier_notes, adviser_ids, contact_ids, interaction_link, policy_area_names, related_trade_agreement_names, policy_issue_type_names, sector, service_delivery, export_barrier_type_names, etc.\n"
                            f"   - For investment-projects-dataset: actual_land_date, address_1, address_2, address_town, address_postcode, anonymous_description, associated_non_fdi_r_and_d_project_id, average_salary__name, client_relationship_manager_id, client_requirements, country_investment_originates_from_id, country_investment_originates_from__name, created_by_id, created_on, description, estimated_land_date, export_revenue, fdi_type__name, fdi_value__name, foreign_equity_investment, government_assistance, gross_value_added, gva_multiplier__multiplier, id, investment_type__name, investor_company_id, investor_type__name, likelihood_to_land__name, modified_by_id, modified_on, name, new_tech_to_uk, non_fdi_r_and_d_budget, number_new_jobs, number_safeguarded_jobs, other_business_activity, project_arrived_in_triage_on, project_assurance_adviser_id, project_manager_id, proposal_deadline, r_and_d_budget, referral_source_activity__name, referral_source_activity_marketing__name, referral_source_activity_website__name, stage__name, status, total_investment, uk_company_id, actual_uk_region_names, business_activity_names, competing_countries, delivery_partner_names, investor_company_sector, level_of_involvement_name, project_first_moved_to_won, project_reference, strategic_driver_names, sector_name, team_member_ids, uk_company_sector, uk_region_location_names, client_contact_ids, client_contact_names, client_contact_emails, specific_programme_names, eyb_lead_ids, etc.\n"
                            f"Use ISO 8601 format for dates. Only include fields that apply to the selected dataset."
                        ),
                    }
                ],
                "max_tokens": 300,
                "temperature": 0.2,
            }
        ),
    )

    body = json.loads(response["body"].read())
    try:
        response_json = json.loads(body["content"][0]["text"].strip())
        return response_json.get("dataset", "companies-dataset"), response_json.get("filters", {})
    except Exception:
        return "companies-dataset", {}


@waffle_flag("DATA_HUB_API_ROUTE_ON")
def filter_results(results, filters):
    def matches(record):
        for key, value in filters.items():
            record_value = record.get(key)
            if record_value is None:
                return False
            elif isinstance(record_value, str) and isinstance(value, str):
                if value.lower() not in record_value.lower():
                    return False
            elif isinstance(record_value, bool):
                if str(record_value).lower() != str(value).lower():
                    return False
            else:
                if value != record_value:
                    return False
        return True

    return [r for r in results if matches(r)]


@waffle_flag("DATA_HUB_API_ROUTE_ON")
def build_search_data_hub_api_tool() -> tool:
    @tool(response_format="content_and_artifact")
    def _search_data_hub(query: str) -> tuple[str, list[Document]]:
        """Search the Data Hub API for relevant datasets based on query."""
        dataset, filters = parse_filters_bedrock(query)

        settings = get_settings()
        base_url = f"{settings.datahub_redbox_url}/v4/dataset/{dataset}"
        secret_key = settings.datahub_redbox_secret_key
        access_key_id = settings.datahub_redbox_access_key_id

        if not base_url or not secret_key or not access_key_id:
            raise ValueError("Data Hub API credentials missing.")

        credentials = {
            "id": access_key_id,
            "key": secret_key,
            "algorithm": "sha256",
        }
        sender = Sender(credentials, base_url, "GET", content="", content_type="")
        headers = {"Authorization": sender.request_header}
        response = requests.get(base_url, headers=headers)
        response.raise_for_status()
        results = response.json()

        if not results or "results" not in results:
            return "No data available for the query.", []

        print(f"Parsed filters: {filters}")

        matches = filter_results(results["results"], filters)

        if not matches:
            return "No matching data found for the query.", []

        mapped_documents = []
        for i, record in enumerate(matches):
            page_content = "\n".join(f"{k}: {v}" for k, v in record.items() if v not in [None, "", []])
            token_count = bedrock_tokeniser(page_content)
            metadata = {
                "index": i,
                "uri": results.get("next", ""),
                "token_count": token_count,
                "creator_type": ChunkCreatorType.data_hub,
            }
            mapped_documents.append(Document(page_content=page_content, metadata=metadata))

        response_content = format_documents(mapped_documents)
        return response_content, mapped_documents

    return _search_data_hub


class BaseRetrievalToolLogFormatter:
    def __init__(self, t: ToolCall) -> None:
        self.tool_call = t

    def log_call(self, tool_call: ToolCall):
        return f"Used {tool_call['name']} to get more information"

    def log_result(self, documents: Iterable[Document]):
        if len(documents) == 0:
            return f"{self.tool_call['name']} returned no documents"
        return f"Reading {documents[1].get('creator_type')} document{'s' if len(documents) > 1 else ''} {','.join(set([d.metadata['uri'].split('/')[-1] for d in documents]))}"


class SearchWikipediaLogFormatter(BaseRetrievalToolLogFormatter):
    def log_call(self):
        return f"Searching Wikipedia for '{self.tool_call['args']['query']}'"

    def log_result(self, documents: Iterable[Document]):
        return f"Reading Wikipedia page{'s' if len(documents) > 1 else ''} {','.join(set([d.metadata['uri'].split('/')[-1] for d in documents]))}"


class SearchDocumentsLogFormatter(BaseRetrievalToolLogFormatter):
    def log_call(self):
        return f"Searching your documents for '{self.tool_call['args']['query']}'"

    def log_result(self, documents: Iterable[Document]):
        return f"Reading {len(documents)} snippets from your documents {','.join(set([d.metadata.get('name', '') for d in documents]))}"


class SearchGovUKLogFormatter(BaseRetrievalToolLogFormatter):
    def log_call(self):
        return f"Searching .gov.uk pages for '{self.tool_call['args']['query']}'"

    def log_result(self, documents: Iterable[Document]):
        return f"Reading pages from .gov.uk, {','.join(set([d.metadata['uri'].split('/')[-1] for d in documents]))}"


@waffle_flag("DATA_HUB_API_ROUTE_ON")
class SearchDataHubLogFormatter(BaseRetrievalToolLogFormatter):
    def log_call(self):
        return f"Searching Data Hub datasets for '{self.tool_call['args']['query']}'"

    def log_result(self, documents: Iterable[Document]):
        if len(documents) == 0:
            return f"{self.tool_call['name']} returned no documents"
        return f"Reading Data Hub dataset document{'s' if len(documents) > 1 else ''} {','.join(set([d.metadata['uri'].split('/')[-1] for d in documents]))}"


__RETRIEVEAL_TOOL_MESSAGE_FORMATTERS = {
    "_search_wikipedia": SearchWikipediaLogFormatter,
    "_search_documents": SearchDocumentsLogFormatter,
    "_search_govuk": SearchGovUKLogFormatter,
}


def get_log_formatter_for_retrieval_tool(t: ToolCall) -> BaseRetrievalToolLogFormatter:
    return __RETRIEVEAL_TOOL_MESSAGE_FORMATTERS.get(t["name"], BaseRetrievalToolLogFormatter)(t)


def web_search_with_retry(
    query: str, no_search_result: int = 20, max_retries: int = 3, country_code: str = "All", ui_lang: str = "en-GB"
) -> requests.Response:
    web_search_settings = get_settings().web_search_settings()
    for attempt in range(max_retries):
        if web_search_settings.name == "Brave":
            response = requests.get(
                web_search_settings.end_point,
                headers=web_search_settings.secret_tokens,
                params={
                    "q": query,
                    "country": country_code,
                    "ui_lang": ui_lang,
                    "count": str(no_search_result),
                    "extra_snippets": "true",
                },
            )
        elif web_search_settings.name == "Kagi":
            response = requests.get(
                web_search_settings.end_point,
                headers=web_search_settings.secret_tokens,
                params={
                    "q": query,
                    "limit": str(no_search_result),
                },
            )
        else:
            log.exception(f"Web search api call to {web_search_settings.name} not currently supported.")
        if response.status_code == 429:
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                delay = (2**attempt) + random.uniform(0, 1)
                time.sleep(delay)
            else:
                log.info("Web search api reach max retry.")
                return response
        else:
            return response


def kagi_response_to_documents(
    tokeniser: Callable, response: requests.Response, mapped_documents: list
) -> list[Document]:
    results = response.json().get("data", [])
    for i, doc in enumerate(results):
        # extract only search results asper kagi documentation
        if doc["t"] == 0:
            mapped_documents = map_documents(tokeniser, i, doc, "snippet", mapped_documents)
    return mapped_documents


def brave_response_to_documents(
    tokeniser: Callable, response: requests.Response, mapped_documents: list
) -> list[Document]:
    results = response.json().get("web", [])
    if type(results) is dict:
        results = results.get("results")
    else:
        results = []
    for i, doc in enumerate(results):
        mapped_documents = map_documents(tokeniser, i, doc, "extra_snippets", mapped_documents)
    return mapped_documents


def map_documents(
    tokeniser: Callable, index: int, doc: str, content_column: str, mapped_documents: list
) -> list[Document]:
    page_content = "".join(doc.get(content_column, []))
    token_count = tokeniser(page_content)
    print(f"Document {index} token count: {token_count}")
    mapped_documents.append(
        Document(
            page_content=page_content,
            metadata=ChunkMetadata(
                index=index,
                uri=doc.get("url", ""),
                token_count=token_count,
                creator_type=ChunkCreatorType.web_search,
            ).model_dump(),
        )
    )
    return mapped_documents


def web_search_call(query: str, no_search_result: int = 20, country_code: str = "All", ui_lang: str = "en-GB") -> tool:
    web_search_settings = get_settings().web_search_settings()
    response = web_search_with_retry(
        query=query,
        no_search_result=no_search_result,
        country_code=country_code,
        ui_lang=ui_lang,
    )
    tokeniser = bedrock_tokeniser
    mapped_documents = []
    if response.status_code == 200:
        if web_search_settings.name == "Brave":
            docs = brave_response_to_documents(tokeniser, response, mapped_documents)
            return format_documents(docs), docs
        elif web_search_settings.name == "Kagi":
            docs = kagi_response_to_documents(tokeniser, response, mapped_documents)
            return format_documents(docs), docs
    else:
        log.exception(f"Web search api call failed. Status: {response.status_code}")
        return "", []


def build_web_search_tool():
    @tool(response_format="content_and_artifact")
    def _search_web(query: str, site: str = ""):
        """
        Web Search tool is a versatile search tool that allows users to search the entire web (similar to a search engine) or to conduct targeted searches within specific websites.

        Args:
            query (str): The search query to pass to web search engine.
            - Can be natural language, keywords, or phrases
            - More specific queries yield more precise results
            - Query length should be 1-500 characters
            site (str, optional): URL or domain to restrict the search to. If provided, results will only be returned from this specific website or domain. If not provided, the search will be performed across all available sources.
        Returns:
            dict[str, Any]: Collection of matching document snippets with metadata:
        """
        if site == "":
            return web_search_call(query=query)
        else:
            return web_search_call(query=query + " site:" + site)

    return _search_web


def build_legislation_search_tool():
    @tool(response_format="content_and_artifact")
    def _search_legislation(query: str):
        """
        Searching legislation.gov.uk.

        Args:
            query (str): The search query to pass to legislation search engine.
            - Can be natural language, keywords, or phrases
            - More specific queries yield more precise results
            - Query length should be 1-500 characters
        Returns:
            dict[str, Any]: Collection of matching document snippets with metadata:
        """
        return web_search_call(query=query + " site:legislation.gov.uk")

    return _search_legislation


def execute_sql_query():
    @tool(response_format="content")
    def _execute_sql_query(sql_query: str, is_intermediate_step: bool, state: Annotated[RedboxState, InjectedState]):
        """
        SQL verification tool is a versatile tool that executes SQL queries against a SQLite database.
        Args:
            sql_query (str): The sql query to be executed against the SQLite database.
            is_intermediate_step (bool): True if your sql query is an intermediate step to allow you to gather information about the database before making the final sql query. False if your sql query would retrieve the relevant information to answer the user question.
        Returns:
            results of the sql query execution if it is successful or an error message if the sql query execution failed
        """
        # execute tabular agent SQL
        conn = sqlite3.connect(state.request.db_location)
        cursor = conn.cursor()
        try:
            cursor.execute(sql_query)
            results = str(cursor.fetchall())
            conn.close()
            if results not in ["", "None", "[]"]:
                return (str(results), "pass", str(is_intermediate_step))
            else:
                error_message = "empty result set. Verify your query."
                return (error_message, "fail", str(is_intermediate_step))
        except Exception as e:
            error_message = (
                f"The SQL query syntax is wrong. Here is the error message: {e}.  Please correct your SQL query."
            )
            return (error_message, "fail", str(is_intermediate_step))

    return _execute_sql_query
