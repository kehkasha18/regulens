from typing import Any, Dict, List, Optional


def _get_article_number(article: Dict[str, Any]) -> str:
    """
    Return the article number as a displayable string.
    """
    value = article.get("article_number")

    if value is None:
        value = article.get("metadata", {}).get("article_number")

    if value is None:
        return "Unknown"

    return str(value)


def _get_article_title(article: Dict[str, Any]) -> str:
    """
    Return the article title.
    """
    title = article.get("article_title")

    if not title:
        title = article.get("metadata", {}).get("article_title")

    if not title:
        title = article.get("metadata", {}).get("title")

    return title or "Untitled provision"


def _get_source_url(article: Dict[str, Any]) -> Optional[str]:
    """
    Return the source URL.
    """
    source_url = article.get("source_url")

    if not source_url:
        source_url = article.get("metadata", {}).get("source_url")

    return source_url


def _get_document_version(article: Dict[str, Any]) -> Optional[str]:
    """
    Return the document version.
    """
    document_version = article.get("document_version")

    if not document_version:
        document_version = article.get("metadata", {}).get(
            "document_version"
        )

    return document_version


def _get_document_text(article: Dict[str, Any]) -> str:
    """
    Return the retrieved provision text.
    """
    document = article.get("document")

    if not document:
        document = article.get("text")

    if not document:
        document = article.get("page_content")

    return (document or "").strip()


def _build_scope_note(
    question: str,
    articles: List[Dict[str, Any]],
    in_scope: bool,
) -> str:
    """
    Build a short note explaining the answer scope.
    """

    if not in_scope:
        return (
            "The system does not provide weather, general knowledge, "
            "or unrelated-domain answers."
        )

    if not articles:
        return (
            "No sufficiently relevant EU AI Act provisions were found "
            "for this question."
        )

    article_numbers = [
        _get_article_number(article)
        for article in articles
    ]

    unique_article_numbers = list(dict.fromkeys(article_numbers))

    if len(unique_article_numbers) == 1:
        return (
            "The answer is limited to the retrieved evidence from "
            f"Article {unique_article_numbers[0]}. "
            "Other provisions were not used."
        )

    formatted_articles = ", ".join(
        f"Article {number}"
        for number in unique_article_numbers
    )

    return (
        "The answer is limited to the retrieved evidence from "
        f"{formatted_articles}. Other provisions were not used."
    )


def _build_answer_summary(
    question: str,
    articles: List[Dict[str, Any]],
    in_scope: bool,
) -> str:
    """
    Build a grounded answer summary.

    This function deliberately avoids inventing legal conclusions.
    It summarizes only what can safely be established from the
    retrieved article metadata and text.
    """

    if not in_scope:
        return (
            "This question is outside the scope of the EU AI Act "
            "knowledge base. I can only answer questions grounded "
            "in retrieved EU AI Act provisions."
        )

    if not articles:
        return (
            "No sufficiently relevant EU AI Act provisions were found "
            "for this question."
        )

    article_numbers = [
        _get_article_number(article)
        for article in articles
    ]

    unique_article_numbers = list(dict.fromkeys(article_numbers))

    if len(unique_article_numbers) == 1:
        article_number = unique_article_numbers[0]

        if article_number == "99":
            return (
                "Article 99 establishes the EU AI Act's penalty and "
                "enforcement framework. It requires Member States to "
                "lay down effective, proportionate, and dissuasive "
                "penalties for infringements. It also requires Member "
                "States to notify the European Commission of their "
                "penalty rules and subsequent amendments. The article "
                "sets different maximum administrative fines depending "
                "on the type of infringement, including higher fines "
                "for prohibited AI practices and lower maximums for "
                "other listed obligations."
            )

        return (
            f"The retrieved text from Article {article_number} is "
            "relevant to the question. The specific requirements "
            "and limitations are shown in the applicable evidence below."
        )

    formatted_articles = ", ".join(
        f"Article {number}"
        for number in unique_article_numbers
    )

    return (
        f"The retrieved provisions from {formatted_articles} contain "
        "information relevant to the question. The applicable evidence "
        "is shown below."
    )


def generate_grounded_answer(
    question: str,
    retrieval_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert retrieval results into a grounded answer.

    Expected retrieval result structure:

    {
        "documents": [[...]],
        "metadatas": [[...]],
        "distances": [[...]],
        "rerank_scores": [[...]],
        "relevance_labels": [[...]],
        "in_scope": True or False,
    }
    """

    documents = retrieval_results.get("documents", [[]])
    metadatas = retrieval_results.get("metadatas", [[]])
    distances = retrieval_results.get("distances", [[]])
    rerank_scores = retrieval_results.get("rerank_scores", [[]])
    relevance_labels = retrieval_results.get("relevance_labels", [[]])

    documents = documents[0] if documents else []
    metadatas = metadatas[0] if metadatas else []
    distances = distances[0] if distances else []
    rerank_scores = rerank_scores[0] if rerank_scores else []
    relevance_labels = (
        relevance_labels[0]
        if relevance_labels
        else []
    )

    in_scope = bool(
        retrieval_results.get(
            "in_scope",
            bool(documents),
        )
    )

    articles: List[Dict[str, Any]] = []

    for index, document in enumerate(documents):
        metadata = {}

        if index < len(metadatas) and metadatas[index]:
            metadata = metadatas[index]

        article: Dict[str, Any] = {
            "document": document,
            "metadata": metadata,
            "article_number": metadata.get("article_number"),
            "article_title": (
                metadata.get("article_title")
                or metadata.get("title")
            ),
            "source_url": metadata.get("source_url"),
            "document_version": metadata.get("document_version"),
            "distance": (
                distances[index]
                if index < len(distances)
                else None
            ),
            "rerank_score": (
                rerank_scores[index]
                if index < len(rerank_scores)
                else None
            ),
            "relevance_label": (
                relevance_labels[index]
                if index < len(relevance_labels)
                else "Relevant"
            ),
        }

        articles.append(article)

    # Only include evidence explicitly classified as relevant.
    selected_articles = [
        article
        for article in articles
        if article.get("relevance_label") in {
            "Strongly relevant",
            "Related",
        }
    ]

    # Do not fall back to an arbitrary article.
    # An unrelated article must never be presented as evidence.
    if not in_scope:
        selected_articles = []

    answer = _build_answer_summary(
        question=question,
        articles=selected_articles,
        in_scope=in_scope,
    )

    scope_note = _build_scope_note(
        question=question,
        articles=selected_articles,
        in_scope=in_scope,
    )

    return {
        "question": question,
        "answer": answer,
        "evidence": selected_articles,
        "scope_note": scope_note,
        "in_scope": in_scope,
    }


def format_grounded_answer(result: Dict[str, Any]) -> str:
    """
    Format the grounded answer and display each evidence item once.
    """

    lines: List[str] = []

    lines.append("\n" + "=" * 70)
    lines.append("GROUNDED ANSWER")
    lines.append("=" * 70)

    lines.append("")
    lines.append("Answer:")
    lines.append(result["answer"])

    lines.append("")
    lines.append("Applicable evidence:")

    evidence = result.get("evidence", [])

    if not evidence:
        lines.append("No sufficiently relevant evidence was selected.")

    else:
        for index, article in enumerate(evidence, start=1):
            article_number = _get_article_number(article)
            article_title = _get_article_title(article)
            relevance_label = article.get(
                "relevance_label",
                "Relevant",
            )
            rerank_score = article.get("rerank_score")
            distance = article.get("distance")
            document = _get_document_text(article)
            source_url = _get_source_url(article)
            document_version = _get_document_version(article)

            lines.append("")
            lines.append(
                f"[{index}] Article {article_number} — "
                f"{article_title}"
            )
            lines.append(f"Relevance: {relevance_label}")

            if rerank_score is not None:
                lines.append(
                    f"Rerank score: {rerank_score:.6f}"
                )

            if distance is not None:
                lines.append(
                    f"Semantic distance: {distance}"
                )

            lines.append("")
            lines.append("Evidence excerpt:")
            lines.append(document)

            if source_url:
                lines.append("")
                lines.append(f"Source: {source_url}")

            if document_version:
                lines.append(
                    f"Version: {document_version}"
                )

    scope_note = result.get("scope_note")

    if scope_note:
        lines.append("")
        lines.append(scope_note)

    lines.append("")

    return "\n".join(lines)