from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from evals.ragas_metrics import RagasMetricScorer
from evals.runner import RagResponse, load_dataset, run_evaluation

REDBOX_ROOT = Path(__file__).resolve().parents[1]
DJANGO_APP_ROOT = REDBOX_ROOT.parent / "django_app"
for import_root in (DJANGO_APP_ROOT, REDBOX_ROOT):
    resolved_root = str(import_root)
    if resolved_root not in sys.path:
        sys.path.insert(0, resolved_root)


def build_runtime(*, skip_ingest: bool):
    from tests.evaluation.eval_pipeline import (
        CORPUS_DIR,
        EVAL_S3_PREFIX,
        build_env,
        build_vector_store,
        cleanup_corpus,
        ingest_corpus,
        make_eval_state,
        make_retriever,
    )

    from redbox.chains.components import get_embeddings
    from redbox.graph.root import get_search_graph
    from redbox.models.chain import RedboxState, AISettings
    from redbox.transform import flatten_document_state

    pdfs = sorted(CORPUS_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No evaluation PDF files found in {CORPUS_DIR} - add evaluation corpus before running RAGAS"
        )

    env = build_env()
    es = env.elasticsearch_client()
    ai_settings = AISettings()
    embeddings = get_embeddings(env)
    vector_store = build_vector_store(env, embeddings)
    retriever = make_retriever(env, es, embeddings)
    search_graph = get_search_graph(retriever=retriever)
    scorer = RagasMetricScorer(
        model=ai_settings.chat_backend.name,
        aws_region=env.aws_region,
        embeddings=embeddings,
    )

    uploaded_keys = []
    cleanup = None
    if skip_ingest:
        uri_map = {pdf.stem: f"{EVAL_S3_PREFIX}/{pdf.name}" for pdf in pdfs}
    else:
        uri_map, uploaded_keys = ingest_corpus(
            env,
            es,
            vector_store,
        )

        def cleanup():
            cleanup_corpus(env, es, uploaded_keys)

    if not uri_map:
        if cleanup is not None:
            cleanup()
        raise ValueError("No documents were ingested into the evaluation corpus - check logs for errors")

    all_uris = list(uri_map.values())

    async def run_question(question: str) -> RagResponse:
        result = await search_graph.ainvoke(make_eval_state(question, all_uris))
        state = RedboxState(**result)
        answer = state.last_message.content
        if not isinstance(answer, str):
            raise TypeError("Redbox returned a non-text answer - RAGAS evaluation requires text")

        contexts = [document.page_content for document in flatten_document_state(state.documents)]
        if not contexts and state.citations:
            contexts = [
                source.highlighted_text_in_source
                for citation in state.citations
                for source in citation.sources
                if source.highlighted_text_in_source.strip()
            ]

        model = (
            state.metadata.llm_calls[-1].llm_model_name
            if state.metadata and state.metadata.llm_calls
            else ai_settings.chat_backend.name
        )
        return RagResponse(answer=answer, contexts=contexts, model=model)

    return run_question, scorer, cleanup


def main() -> None:
    default_dataset = Path(__file__).with_name("export_support.yaml")
    default_output_dir = Path(__file__).with_name("results")

    parser = argparse.ArgumentParser(description="Run RAGAS quality evaluation against live service")
    parser.add_argument("--dataset", type=Path, default=default_dataset, help="YAML eval question set")
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_dir / "ragas_results.jsonl",
        help="Append per question records to this jsonl file",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=default_output_dir / "ragas_report_latest.html",
        help="Write an HTML report for this run",
    )
    parser.add_argument("--skip-ingest", action="store_true", help="use existing evaluation index")
    parser.add_argument("--no-cleanup", action="store_true", help="Leave evaluation index and uploaded files after run")
    args = parser.parse_args()

    service, prompt_version, cases = load_dataset(args.dataset)
    rag_runner, metric_scorer, cleanup = build_runtime(skip_ingest=args.skip_ingest)
    try:
        records = asyncio.run(
            run_evaluation(
                cases=cases,
                service=service,
                prompt_version=prompt_version,
                env=os.getenv("ENVIRONMENT", "LOCAL").upper(),
                rag_runner=rag_runner,
                metric_scorer=metric_scorer,
                output_path=args.output,
                report_path=args.report,
            )
        )
    finally:
        if cleanup is not None and not args.no_cleanup:
            cleanup()
    print(f"Evaluated {len(records)} questions. JSONL: {args.output}, HTML report: {args.report}")


if __name__ == "__main__":
    main()
