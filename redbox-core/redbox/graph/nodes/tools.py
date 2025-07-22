from typing import Annotated, Iterable, Union, Tuple

import aiohttp
import boto3
import json
import logging
import numpy as np
import requests
from elasticsearch import Elasticsearch
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.documents import Document
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.messages import ToolCall
from langchain_core.tools import Tool, tool, StructuredTool
from langgraph.prebuilt import InjectedState
from mohawk import Sender
from opensearchpy import OpenSearch
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity
from waffle.decorators import waffle_flag

from redbox.api.format import format_documents
from redbox.chains.components import get_embeddings
from redbox.models.chain import RedboxState
from redbox.models.file import ChunkCreatorType, ChunkMetadata, ChunkResolution
from redbox.models.settings import get_settings
from redbox.retriever.queries import add_document_filter_scores_to_query, build_document_query
from redbox.retriever.retrievers import query_to_documents
from redbox.transform import bedrock_tokeniser, merge_documents, sort_documents


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

        # Merge and sort
        merged_documents = merge_documents(initial=initial_documents, adjacent=adjacent_boosted)
        sorted_documents = sort_documents(documents=merged_documents)

        # Return as state update
        return format_documents(sorted_documents), sorted_documents

    return _search_documents


log = logging.getLogger(__name__)


class GovUKSearchInput(BaseModel):
    query: str = Field(description="The search query string to match against gov.uk content")
    state: Annotated[RedboxState, InjectedState] = Field(description="The current state of the application")

def build_govuk_search_tool(filter: bool = True) -> StructuredTool:
    """
    Constructs a tool that asynchronously searches gov.uk and returns formatted documents and raw documents.
    """
    @tool(response_format="content_and_artifact")
    async def _search_govuk(query: str, state: Annotated[RedboxState, InjectedState]) -> Tuple[str, list[Document]]:
        """
        Search for documents on gov.uk based on a query string.
        This endpoint is used to search for documents on gov.uk. There are many types of documents on gov.uk.
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
        """
        tokeniser = bedrock_tokeniser
        max_content_tokens = 1000
        url_base = "https://www.gov.uk"
        required_fields = ["format", "title", "description", "indexable_content", "link"]
        ai_settings = state.request.ai_settings
        timeout = aiohttp.ClientTimeout(total=30)

        async def recalculate_similarity(response: dict, query: str, num_results: int) -> dict:
            try:
                embedding_model = get_embeddings(get_settings())
                query_vector = embedding_model.embed_query(query)
                for result in response.get("results", []):
                    description = result.get("description", "")
                    if not description:
                        result["similarity"] = 0.0
                        continue
                    description_vector = embedding_model.embed_query(description)
                    result["similarity"] = cosine_similarity(
                        np.array(query_vector).reshape(1, -1), np.array(description_vector).reshape(1, -1)
                    )[0][0]
                response["results"] = sorted(response.get("results", []), key=lambda x: x["similarity"], reverse=True)[:num_results]
                return response
            except Exception as e:
                log.error(f"Error in recalculate_similarity: {str(e)}")
                return response

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    f"{url_base}/api/search.json",
                    params={
                        "q": query,
                        "count": (
                            ai_settings.tool_govuk_retrieved_results
                            if filter
                            else ai_settings.tool_govuk_returned_results
                        ),
                        "fields": required_fields,
                    },
                    headers={"Accept": "application/json"},
                ) as response:
                    response.raise_for_status()
                    try:
                        response_data = await response.json()
                        log.debug(f"Raw gov.uk API response: {response_data}")
                    except aiohttp.ContentTypeError as e:
                        log.error(f"Invalid content type from gov.uk API: {str(e)}")
                        return "Error: Invalid response format from gov.uk API", []
                    except ValueError as e:
                        log.error(f"Failed to parse JSON from gov.uk API: {str(e)}")
                        return "Error: Failed to parse API response", []

            if not response_data.get("results"):
                log.warning(f"No results found for query: {query}")
                return "No results found from gov.uk API.", []

            if filter:
                response_data = await recalculate_similarity(
                    response_data, query, ai_settings.tool_govuk_returned_results
                )

            mapped_documents = []
            for i, doc in enumerate(response_data.get("results", [])):
                if any(field not in doc for field in required_fields):
                    log.warning(f"Skipping document {i} due to missing required fields: {doc}")
                    continue
                content = doc["indexable_content"][:max_content_tokens]
                if not content:
                    log.warning(f"Skipping document {i} due to empty content")
                    continue
                try:
                    token_count = tokeniser(content)
                    mapped_documents.append(
                        Document(
                            page_content=content,
                            metadata=ChunkMetadata(
                                index=i,
                                uri=f"{url_base}{doc['link']}",
                                token_count=token_count,
                                creator_type=ChunkCreatorType.gov_uk,
                            ).model_dump(),
                        )
                    )
                except Exception as e:
                    log.warning(f"Error creating Document for gov.uk result {i}: {str(e)}")
                    continue

            if not mapped_documents:
                log.warning(f"No valid documents after processing for query: {query}")
                return "No valid documents found from gov.uk API.", []

            formatted_content = format_documents(mapped_documents)
            log.debug(f"Formatted content: {formatted_content[:500]}...")
            return formatted_content, mapped_documents

        except aiohttp.ClientError as e:
            log.error(f"Error accessing gov.uk API: {str(e)}")
            return f"Error accessing gov.uk API: {str(e)}", []
        except Exception as e:
            log.error(f"Unexpected error in _search_govuk: {str(e)}")
            return f"Error: {str(e)}", []

    return StructuredTool.from_function(
        func=_search_govuk,
        name="_search_govuk",
        description="Asynchronously search gov.uk for relevant documents.",
        args_schema=GovUKSearchInput,
        response_format="content_and_artifact",
        coroutine=_search_govuk,
    )


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
            tuple[str, list[Document]]: Formatted content and list of Document objects
        """
        log.debug(f"Executing Wikipedia search for query: {query}")
        response = _wikipedia_wrapper.load(query)
        log.debug(f"Wikipedia API response: {response}")
        if not response:
            log.warning(f"No Wikipedia response found for query: {query}")
            return "", []

        mapped_documents = []
        for i, doc in enumerate(response):
            if not hasattr(doc, "page_content") or not hasattr(doc, "metadata"):
                log.warning(f"Invalid document from Wikipedia at index {i}: {doc}")
                continue
            token_count = tokeniser(doc.page_content)
            log.debug(f"Document {i} token count: {token_count}")
            mapped_documents.append(
                Document(
                    page_content=doc.page_content[:max_chars_per_wiki_page],
                    metadata=ChunkMetadata(
                        index=i,
                        uri=doc.metadata.get("source", ""),
                        token_count=token_count,
                        creator_type=ChunkCreatorType.wikipedia,
                    ).model_dump(),
                )
            )
        if not mapped_documents:
            log.warning(f"No valid documents after processing for query: {query}")
            return "No valid Wikipedia pages found.", []
        formatted_content = format_documents(mapped_documents)
        log.debug(f"Formatted content: {formatted_content[:500]}...")
        return formatted_content, mapped_documents

    return _search_wikipedia


@waffle_flag("DATA_HUB_API_ROUTE_ON")
def parse_filters_bedrock(prompt: str):
    client = boto3.client("bedrock-runtime", region_name="eu-west-2")
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

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
        return f"Used {tool_call["name"]} to get more information"

    def log_result(self, documents: Iterable[Document]):
        if len(documents) == 0:
            return f"{self.tool_call["name"]} returned no documents"
        return f"Reading {documents[1].get("creator_type")} document{"s" if len(documents)>1 else ""} {','.join(set([d.metadata["uri"].split("/")[-1] for d in documents]))}"


class SearchWikipediaLogFormatter(BaseRetrievalToolLogFormatter):
    def log_call(self):
        return f"Searching Wikipedia for '{self.tool_call["args"]["query"]}'"

    def log_result(self, documents: Iterable[Document]):
        return f"Reading Wikipedia page{"s" if len(documents)>1 else ""} {','.join(set([d.metadata["uri"].split("/")[-1] for d in documents]))}"


class SearchDocumentsLogFormatter(BaseRetrievalToolLogFormatter):
    def log_call(self):
        return f"Searching your documents for '{self.tool_call["args"]["query"]}'"

    def log_result(self, documents: Iterable[Document]):
        return f"Reading {len(documents)} snippets from your documents {','.join(set([d.metadata.get("name", "") for d in documents]))}"


class SearchGovUKLogFormatter(BaseRetrievalToolLogFormatter):
    def log_call(self):
        return f"Searching .gov.uk pages for '{self.tool_call["args"]["query"]}'"

    def log_result(self, documents: Iterable[Document]):
        return f"Reading pages from .gov.uk, {','.join(set([d.metadata["uri"].split("/")[-1] for d in documents]))}"


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
