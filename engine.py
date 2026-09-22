from pathlib import Path
import re
from typing import Any, Dict, List, Set

import chromadb
from chromadb.utils import embedding_functions

from answer_generator import (
    generate_grounded_answer,
    format_grounded_answer,
)


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# These values must match the existing indexed database.
DB_PATH = BASE_DIR / "data" / "compliance_db"
COLLECTION_NAME = "regulations"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

DEFAULT_RESULTS = 3
MIN_CANDIDATES = 20


# ============================================================
# Embedding model and Chroma collection
# ============================================================

print("Loading embedding model...")

embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

print("Connecting to Chroma database...")

client = chromadb.PersistentClient(
    path=str(DB_PATH)
)

collection = client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_model,
)


# ============================================================
# Basic text processing
# ============================================================

STOPWORDS: Set[str] = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "say",
    "should",
    "that",
    "the",
    "their",
    "this",
    "to",
    "under",
    "use",
    "what",
    "when",
    "with",
    "would",
    "you",
}


def tokenize(text: str) -> Set[str]:
    """
    Convert text into normalized keyword tokens.
    """
    if not text:
        return set()

    tokens = re.findall(
        r"[a-z0-9]+",
        text.lower(),
    )

    return {
        token
        for token in tokens
        if token not in STOPWORDS and len(token) > 2
    }


def normalize_text(text: str) -> str:
    """
    Normalize whitespace and casing for phrase matching.
    """
    if not text:
        return ""

    return re.sub(
        r"\s+",
        " ",
        text.lower(),
    ).strip()


def clean_question(question: str) -> str:
    """
    Prevent duplicated prefixes such as:

        Question: Question: What is human oversight?
    """
    cleaned = str(question or "").strip()

    while cleaned.lower().startswith("question:"):
        cleaned = cleaned.split(":", 1)[1].strip()

    return cleaned


# ============================================================
# Scope detection
# ============================================================

EU_AI_ACT_PHRASES = {
    "eu ai act",
    "artificial intelligence act",
    "regulation 2024/1689",
    "regulation (eu) 2024/1689",
    "32024r1689",
    "high risk ai",
    "high-risk ai",
    "general purpose ai",
    "general-purpose ai",
    "ai system",
    "ai systems",
    "ai model",
    "ai models",
    "ai provider",
    "ai providers",
    "ai deployer",
    "ai deployers",
    "ai literacy",
    "human oversight",
    "human supervision",
    "risk management system",
    "technical documentation",
    "conformity assessment",
    "market surveillance",
    "notified body",
    "notified bodies",
    "ai governance",
    "ai regulation",
    "ai compliance",
    "prohibited ai practices",
    "transparency obligations",
    "fundamental rights impact assessment",
    "regulatory sandbox",
}

EU_AI_ACT_LEGAL_TERMS = {
    "obligation",
    "obligations",
    "requirement",
    "requirements",
    "compliance",
    "regulation",
    "regulations",
    "article",
    "articles",
    "chapter",
    "section",
    "penalty",
    "penalties",
    "fine",
    "fines",
    "prohibition",
    "prohibitions",
    "provider",
    "providers",
    "deployer",
    "deployers",
    "operator",
    "operators",
    "authority",
    "authorities",
    "risk",
    "transparency",
    "oversight",
    "supervision",
    "documentation",
    "assessment",
    "governance",
    "enforcement",
    "classification",
    "general purpose",
    "high risk",
}


def extract_requested_article_numbers(question: str) -> Set[int]:
    """
    Extract explicit references such as:
    Article 5
    Article 99
    Articles 6 and 7
    Article 9(2)
    """
    normalized = normalize_text(question)

    matches = re.findall(
        r"\barticles?\s+(\d+)",
        normalized,
    )

    return {int(number) for number in matches}


def is_in_scope_question(question: str) -> bool:
    """
    Reject unrelated questions before semantic retrieval.

    For example:
    - How is the weather in Delhi? -> False
    - What are the obligations under Article 99? -> True
    - What is human oversight? -> True
    - What are transparency obligations? -> True
    """
    normalized = normalize_text(question)

    if not normalized:
        return False

    for phrase in EU_AI_ACT_PHRASES:
        if phrase in normalized:
            return True

    article_numbers = extract_requested_article_numbers(normalized)

    if article_numbers:
        if len(normalized.split()) <= 5:
            return True

        if any(
            term in normalized
            for term in EU_AI_ACT_LEGAL_TERMS
        ):
            return True

    has_ai_reference = bool(
        re.search(r"\bai\b", normalized)
        or "artificial intelligence" in normalized
    )

    has_legal_reference = any(
        term in normalized
        for term in EU_AI_ACT_LEGAL_TERMS
    )

    if has_ai_reference and has_legal_reference:
        return True

    return False


# ============================================================
# Domain concept expansion
# ============================================================

CONCEPT_GROUPS: Dict[str, Set[str]] = {
    "human_oversight": {
        "human",
        "oversight",
        "oversee",
        "supervise",
        "supervision",
        "monitor",
        "monitoring",
        "intervene",
        "intervention",
        "override",
        "stop",
        "review",
        "control",
    },
    "transparency": {
        "transparency",
        "transparent",
        "inform",
        "information",
        "disclose",
        "disclosure",
        "notice",
        "label",
        "labelling",
        "labeling",
        "interaction",
        "deepfake",
        "machine",
        "readable",
    },
    "employment": {
        "employment",
        "employer",
        "employee",
        "worker",
        "workers",
        "workplace",
        "hiring",
        "hire",
        "recruitment",
        "recruit",
        "recruiting",
        "candidate",
        "candidates",
        "resume",
        "resumes",
        "curriculum",
        "vitae",
        "cv",
        "job",
        "jobs",
        "selection",
        "promotion",
        "termination",
        "performance",
        "appraisal",
    },
    "high_risk": {
        "high",
        "risk",
        "risky",
        "classification",
        "classify",
        "system",
        "systems",
    },
    "penalties": {
        "penalty",
        "penalties",
        "fine",
        "fines",
        "administrative",
        "sanction",
        "sanctions",
        "enforcement",
    },
}


def expand_query_terms(query: str) -> Set[str]:
    """
    Expand query terms when the query clearly belongs to a
    relevant regulatory concept.
    """
    query_terms = tokenize(query)
    expanded_terms = set(query_terms)

    for concept_terms in CONCEPT_GROUPS.values():
        if query_terms.intersection(concept_terms):
            expanded_terms.update(concept_terms)

    return expanded_terms


# ============================================================
# Relevance scoring
# ============================================================

def overlap_score(
    query_terms: Set[str],
    document_terms: Set[str],
) -> float:
    """
    Calculate normalized token overlap.
    """
    if not query_terms:
        return 0.0

    return len(
        query_terms.intersection(document_terms)
    ) / len(query_terms)


def calculate_rerank_score(
    query: str,
    document: str,
    metadata: Dict[str, Any],
    semantic_distance: float,
) -> Dict[str, float]:
    """
    Combine semantic similarity and lexical relevance.
    """
    title = str(
        metadata.get(
            "article_title",
            metadata.get("title", ""),
        )
    )

    query_normalized = normalize_text(query)
    document_normalized = normalize_text(document)
    title_normalized = normalize_text(title)

    query_terms = tokenize(query)
    expanded_terms = expand_query_terms(query)

    title_terms = tokenize(title)
    document_terms = tokenize(document)

    try:
        distance = max(float(semantic_distance), 0.0)
    except (TypeError, ValueError):
        distance = 1.0

    semantic_score = 1.0 / (1.0 + distance)

    exact_phrase_score = 0.0

    if len(query_normalized) >= 12:
        if query_normalized in title_normalized:
            exact_phrase_score = 1.0
        elif query_normalized in document_normalized:
            exact_phrase_score = 0.75

    title_overlap = overlap_score(
        query_terms,
        title_terms,
    )

    direct_body_overlap = overlap_score(
        query_terms,
        document_terms,
    )

    expanded_body_overlap = overlap_score(
        expanded_terms,
        document_terms,
    )

    rerank_score = (
        0.45 * semantic_score
        + 0.25 * title_overlap
        + 0.15 * direct_body_overlap
        + 0.10 * expanded_body_overlap
        + 0.05 * exact_phrase_score
    )

    return {
        "rerank_score": round(rerank_score, 6),
        "semantic_score": round(semantic_score, 6),
        "title_overlap": round(title_overlap, 6),
        "body_overlap": round(direct_body_overlap, 6),
        "expanded_body_overlap": round(
            expanded_body_overlap,
            6,
        ),
        "exact_phrase_score": round(
            exact_phrase_score,
            6,
        ),
    }


# ============================================================
# Exact article matching
# ============================================================

def extract_article_number(
    metadata: Dict[str, Any],
    document: str,
) -> int | None:
    """
    Extract article number from metadata first, then document text.
    """
    for key in (
        "article_number",
        "article",
        "article_no",
        "article_id",
    ):
        value = metadata.get(key)

        if value is not None:
            match = re.search(
                r"\d+",
                str(value),
            )

            if match:
                return int(match.group())

    match = re.search(
        r"\barticle\s+(\d+)\b",
        normalize_text(document),
    )

    if match:
        return int(match.group(1))

    return None


def matches_requested_articles(
    metadata: Dict[str, Any],
    document: str,
    requested_articles: Set[int],
) -> bool:
    """
    Ensure Article 99 questions only retrieve Article 99.
    """
    if not requested_articles:
        return True

    article_number = extract_article_number(
        metadata,
        document,
    )

    return article_number in requested_articles


# ============================================================
# Retrieval and reranking
# ============================================================

def query_compliance_engine(
    query_text: str,
    n_results: int = DEFAULT_RESULTS,
) -> Dict[str, Any]:
    """
    Retrieve and rerank EU AI Act provisions.

    The returned structure remains compatible with the existing
    answer_generator.py and Streamlit application.
    """
    query_text = clean_question(query_text)

    empty_result = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
        "rerank_scores": [[]],
        "relevance_labels": [[]],
        "in_scope": False,
        "requested_articles": [],
    }

    if not query_text:
        return empty_result

    requested_articles = extract_requested_article_numbers(
        query_text
    )

    empty_result["requested_articles"] = sorted(
        requested_articles
    )

    # --------------------------------------------------------
    # Step 1: Scope gate
    # --------------------------------------------------------

    if not is_in_scope_question(query_text):
        return empty_result

    empty_result["in_scope"] = True

    # --------------------------------------------------------
    # Step 2: Check database
    # --------------------------------------------------------

    total_documents = collection.count()

    if total_documents == 0:
        raise RuntimeError(
            "The compliance collection is empty. "
            "Please run ingest.py first."
        )

    requested_results = max(
        1,
        int(n_results),
    )

    # For explicit article questions, inspect the complete
    # collection so the requested article is not missed.
    if requested_articles:
        candidate_count = total_documents
    else:
        candidate_count = min(
            total_documents,
            max(
                MIN_CANDIDATES,
                requested_results * 6,
            ),
        )

    # --------------------------------------------------------
    # Step 3: Chroma semantic retrieval
    # --------------------------------------------------------

    raw_results = collection.query(
        query_texts=[query_text],
        n_results=candidate_count,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    raw_ids = raw_results.get("ids", [[]])[0]
    raw_documents = raw_results.get("documents", [[]])[0]
    raw_metadatas = raw_results.get("metadatas", [[]])[0]
    raw_distances = raw_results.get("distances", [[]])[0]

    query_terms = tokenize(query_text)
    expanded_terms = expand_query_terms(query_text)

    ranked_records: List[Dict[str, Any]] = []

    # --------------------------------------------------------
    # Step 4: Exact article filtering and reranking
    # --------------------------------------------------------

    for index, document in enumerate(raw_documents):
        if not document:
            continue

        metadata = (
            raw_metadatas[index]
            if index < len(raw_metadatas)
            and raw_metadatas[index]
            else {}
        )

        distance = (
            raw_distances[index]
            if index < len(raw_distances)
            else 1.0
        )

        # Critical protection:
        # Article 99 must not retrieve Article 16, Article 22, etc.
        if not matches_requested_articles(
            metadata,
            document,
            requested_articles,
        ):
            continue

        score_details = calculate_rerank_score(
            query=query_text,
            document=document,
            metadata=metadata,
            semantic_distance=distance,
        )

        title = str(
            metadata.get(
                "article_title",
                metadata.get("title", ""),
            )
        )

        title_terms = tokenize(title)
        document_terms = tokenize(document)

        direct_title_overlap = overlap_score(
            query_terms,
            title_terms,
        )

        direct_body_overlap = overlap_score(
            query_terms,
            document_terms,
        )

        expanded_body_overlap = overlap_score(
            expanded_terms,
            document_terms,
        )

        meaningful_lexical_signal = max(
            direct_title_overlap,
            direct_body_overlap,
            expanded_body_overlap,
        )

        score_details["meaningful_lexical_signal"] = round(
            meaningful_lexical_signal,
            6,
        )

        # ----------------------------------------------------
        # Conservative relevance labels
        # ----------------------------------------------------

        exact_article_match = bool(requested_articles)

        if exact_article_match:
            relevance_label = "Strongly relevant"

        elif (
            direct_title_overlap >= 0.30
            and score_details["rerank_score"] >= 0.48
        ):
            relevance_label = "Strongly relevant"

        elif (
            meaningful_lexical_signal >= 0.15
            and score_details["rerank_score"] >= 0.44
        ):
            relevance_label = "Related"

        else:
            relevance_label = "Weak match"

        ranked_records.append(
            {
                "id": (
                    raw_ids[index]
                    if index < len(raw_ids)
                    else None
                ),
                "document": document,
                "metadata": metadata,
                "distance": distance,
                "relevance_label": relevance_label,
                **score_details,
            }
        )

    ranked_records.sort(
        key=lambda record: record["rerank_score"],
        reverse=True,
    )

    # --------------------------------------------------------
    # Step 5: Remove weak matches
    # --------------------------------------------------------

    meaningful_records = [
        record
        for record in ranked_records
        if record["relevance_label"]
        in {
            "Strongly relevant",
            "Related",
        }
    ]

    selected_records = meaningful_records[
        :requested_results
    ]

    # --------------------------------------------------------
    # Step 6: Return Chroma-compatible structure
    # --------------------------------------------------------

    return {
        "ids": [
            [
                record["id"]
                for record in selected_records
            ]
        ],
        "documents": [
            [
                record["document"]
                for record in selected_records
            ]
        ],
        "metadatas": [
            [
                record["metadata"]
                for record in selected_records
            ]
        ],
        "distances": [
            [
                record["distance"]
                for record in selected_records
            ]
        ],
        "rerank_scores": [
            [
                record["rerank_score"]
                for record in selected_records
            ]
        ],
        "relevance_labels": [
            [
                record["relevance_label"]
                for record in selected_records
            ]
        ],
        "in_scope": True,
        "requested_articles": sorted(
            requested_articles
        ),
    }


# ============================================================
# Retrieval result display
# ============================================================

def print_results(
    question: str,
    results: Dict[str, Any],
) -> None:
    """
    Display retrieved provisions in a readable format.
    """
    documents = results.get(
        "documents",
        [[]],
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]],
    )[0]

    distances = results.get(
        "distances",
        [[]],
    )[0]

    rerank_scores = results.get(
        "rerank_scores",
        [[]],
    )[0]

    relevance_labels = results.get(
        "relevance_labels",
        [[]],
    )[0]

    print("\n" + "=" * 70)
    print(f"Retrieved evidence for: {question}")
    print("=" * 70)

    if not documents:
        print("No relevant EU AI Act provisions were found.")
        return

    for index, document in enumerate(documents):
        metadata = (
            metadatas[index]
            if index < len(metadatas)
            else {}
        ) or {}

        article_number = metadata.get(
            "article_number",
            metadata.get("article", "Unknown"),
        )

        article_title = metadata.get(
            "article_title",
            metadata.get("title", "Untitled"),
        )

        source_url = metadata.get(
            "source_url",
            "",
        )

        document_version = metadata.get(
            "document_version",
            "",
        )

        distance = (
            distances[index]
            if index < len(distances)
            else None
        )

        rerank_score = (
            rerank_scores[index]
            if index < len(rerank_scores)
            else None
        )

        relevance_label = (
            relevance_labels[index]
            if index < len(relevance_labels)
            else "Unclassified"
        )

        print(
            f"\n[{index + 1}] Article {article_number} — "
            f"{article_title}"
        )

        print(f"Relevance: {relevance_label}")

        if rerank_score is not None:
            print(f"Rerank score: {rerank_score}")

        if distance is not None:
            print(f"Semantic distance: {distance}")

        if document_version:
            print(f"Version: {document_version}")

        if source_url:
            print(f"Source: {source_url}")

        print("\nProvision excerpt:")
        print(document[:1800])

        if len(document) > 1800:
            print("...")


# ============================================================
# Complete terminal workflow
# ============================================================

def main() -> None:
    total_documents = collection.count()

    print("=" * 70)
    print("ReguLens — Evidence-Grounded Regulatory Intelligence")
    print("=" * 70)
    print(f"Knowledge base: {total_documents} indexed provisions")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Embedding model: {EMBEDDING_MODEL}")

    if total_documents == 0:
        print(
            "\nThe knowledge base is empty. "
            "Run ingest.py before using the engine."
        )
        return

    print("\nAsk questions about the EU AI Act.")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        try:
            question = input("\nQuestion: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not question:
            continue

        if question.lower() in {
            "exit",
            "quit",
        }:
            print("Exiting.")
            break

        cleaned_question = clean_question(question)

        try:
            results = query_compliance_engine(
                cleaned_question,
                n_results=DEFAULT_RESULTS,
            )


            grounded_result = generate_grounded_answer(
                question=cleaned_question,
                retrieval_results=results,
            )

            print(
                format_grounded_answer(
                    grounded_result
                )
            )

        except Exception as error:
            print(f"\nError: {error}")


if __name__ == "__main__":
    main()