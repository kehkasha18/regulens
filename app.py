import streamlit as st

from engine import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    DEFAULT_RESULTS,
    clean_question,
    collection,
    query_compliance_engine,
)

from answer_generator import generate_grounded_answer


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="ReguLens",
    page_icon="⚖️",
    layout="wide",
)


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #6b7280;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .metric-card {
            padding: 1rem;
            border: 1px solid #e5e7eb;
            border-radius: 0.75rem;
            background-color: #f9fafb;
        }

        .answer-box {
            padding: 1.2rem;
            border-left: 4px solid #2563eb;
            border-radius: 0.5rem;
            background-color: #f8fafc;
            margin-bottom: 1rem;
        }

        .evidence-box {
            padding: 1rem;
            border: 1px solid #e5e7eb;
            border-radius: 0.6rem;
            background-color: #ffffff;
            margin-bottom: 1rem;
        }

        .small-label {
            color: #6b7280;
            font-size: 0.85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def display_evidence(evidence):
    """
    Display retrieved evidence in expandable sections.
    """

    if not evidence:
        st.info("No sufficiently relevant evidence was selected.")
        return

    for index, article in enumerate(evidence, start=1):
        article_number = article.get(
            "article_number",
            article.get("metadata", {}).get("article_number", "Unknown"),
        )

        article_title = article.get(
            "article_title"
        ) or article.get(
            "metadata", {}
        ).get(
            "article_title",
            article.get("metadata", {}).get("title", "Untitled provision"),
        )

        relevance_label = article.get(
            "relevance_label",
            "Relevant",
        )

        rerank_score = article.get("rerank_score")
        distance = article.get("distance")

        document = (
            article.get("document")
            or article.get("text")
            or article.get("page_content")
            or ""
        )

        source_url = article.get(
            "source_url"
        ) or article.get(
            "metadata", {}
        ).get("source_url")

        document_version = article.get(
            "document_version"
        ) or article.get(
            "metadata", {}
        ).get("document_version")

        heading = (
            f"Article {article_number} — "
            f"{article_title}"
        )

        with st.expander(
            f"{index}. {heading}",
            expanded=True,
        ):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(
                    f"**Relevance:** {relevance_label}"
                )

            with col2:
                if rerank_score is not None:
                    st.markdown(
                        f"**Rerank score:** {rerank_score:.6f}"
                    )

            with col3:
                if distance is not None:
                    st.markdown(
                        f"**Semantic distance:** {distance}"
                    )

            st.markdown("**Evidence excerpt**")
            st.text_area(
                label=f"Evidence excerpt {index}",
                value=document,
                height=280,
                label_visibility="collapsed",
                disabled=True,
            )

            if source_url:
                st.markdown(
                    f"**Source:** [{source_url}]({source_url})"
                )

            if document_version:
                st.markdown(
                    f"**Version:** {document_version}"
                )


def run_query(question: str):
    """
    Run the existing ReguLens retrieval and answer pipeline.
    """

    cleaned_question = clean_question(question)

    results = query_compliance_engine(
        cleaned_question,
        n_results=DEFAULT_RESULTS,
    )

    grounded_result = generate_grounded_answer(
        question=cleaned_question,
        retrieval_results=results,
    )

    return grounded_result


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.header("ReguLens")

    st.markdown(
        "Evidence-grounded regulatory intelligence "
        "for the EU AI Act."
    )

    st.divider()

    st.subheader("Knowledge base")

    total_documents = collection.count()

    st.metric(
        label="Indexed provisions",
        value=total_documents,
    )

    st.caption(
        f"Collection: {COLLECTION_NAME}"
    )

    st.caption(
        f"Embedding model: {EMBEDDING_MODEL}"
    )

    st.divider()

    st.subheader("Example questions")

    example_questions = [
        "What responsibilities does an organization have when deploying a high-risk AI system?",
        "How should human oversight be handled for high-risk AI systems?",
        "What are the main obligations for providers of high-risk AI systems?",
        "What information must people receive when they interact with certain AI systems?",
        "How does the EU AI Act address fines and penalties for non-compliance?",
    ]

    for example in example_questions:
        if st.button(
            example,
            use_container_width=True,
        ):
            st.session_state.question = example


# ---------------------------------------------------------
# Main interface
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">⚖️ ReguLens</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Evidence-Grounded Regulatory Intelligence for the EU AI Act
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Ask a question about the EU AI Act. "
    "ReguLens answers only from retrieved regulatory evidence "
    "and rejects unrelated questions."
)


if "question" not in st.session_state:
    st.session_state.question = ""


question = st.text_area(
    "Enter your question",
    value=st.session_state.question,
    placeholder="Example: What are the obligations under Article 99?",
    height=100,
)

ask_button = st.button(
    "🔍 Ask ReguLens",
    type="primary",
    use_container_width=False,
)


if ask_button:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving and analyzing regulatory evidence..."):
            try:
                result = run_query(question)

                st.session_state.last_result = result

            except Exception as error:
                st.error(
                    f"An error occurred while processing the question: {error}"
                )


# ---------------------------------------------------------
# Display result
# ---------------------------------------------------------

if "last_result" in st.session_state:
    result = st.session_state.last_result

    st.divider()

    st.subheader("Grounded answer")

    st.markdown(
        f"""
        <div class="answer-box">
            {result["answer"]}
        </div>
        """,
        unsafe_allow_html=True,
    )

    evidence = result.get("evidence", [])

    if evidence:
        st.subheader("Applicable evidence")
        display_evidence(evidence)
    else:
        st.info(
            "No sufficiently relevant evidence was selected."
        )

    scope_note = result.get("scope_note")

    if scope_note:
        st.caption(scope_note)