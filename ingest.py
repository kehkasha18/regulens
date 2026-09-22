
from pathlib import Path
import re
import shutil

from bs4 import BeautifulSoup
import chromadb
from chromadb.utils import embedding_functions


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
HTML_PATH = BASE_DIR / "data" / "eu_ai_act.html"
DB_PATH = BASE_DIR / "data" / "compliance_db"

COLLECTION_NAME = "regulations"

SOURCE_URL = (
    "https://eur-lex.europa.eu/legal-content/EN/TXT/"
    "?uri=CELEX:32024R1689"
)

DOCUMENT_VERSION = "Current consolidated version: 27/07/2026"


# --------------------------------------------------
# Read and clean HTML
# --------------------------------------------------

def load_html_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"HTML file not found: {path}")

    html = path.read_text(encoding="utf-8", errors="ignore")

    if not html.strip():
        raise ValueError("The HTML file is empty.")

    soup = BeautifulSoup(html, "html.parser")

    # Remove page elements that are not regulatory content.
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text("\n")

    # Normalize whitespace while preserving line boundaries.
    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)

    return "\n".join(lines)


# --------------------------------------------------
# Extract articles
# --------------------------------------------------

def extract_articles(text: str) -> list[dict]:
    """
    Extract article sections from the official HTML text.

    Expected structure:
        Article 1
        Subject matter
        ...
        Article 2
        Definitions
        ...
    """

    lines = text.splitlines()

    article_pattern = re.compile(
        r"^Article\s+(\d+)\s*$",
        re.IGNORECASE
    )

    article_starts = []

    for index, line in enumerate(lines):
        match = article_pattern.match(line)
        if match:
            article_starts.append(
                (index, int(match.group(1)))
            )

    if not article_starts:
        raise ValueError(
            "No article headings were detected. "
            "The HTML structure may need inspection."
        )

    articles = []

    for position, (start_index, article_number) in enumerate(article_starts):
        end_index = (
            article_starts[position + 1][0]
            if position + 1 < len(article_starts)
            else len(lines)
        )

        section_lines = lines[start_index:end_index]

        # Avoid accidentally including enormous unrelated sections.
        section_text = "\n".join(section_lines).strip()

        if len(section_text) < 40:
            continue

        # Usually the title is the line immediately after "Article N".
        title = (
            section_lines[1]
            if len(section_lines) > 1
            else f"Article {article_number}"
        )

        articles.append(
            {
                "article_number": article_number,
                "title": title,
                "text": section_text,
            }
        )

    # Remove duplicate article numbers while preserving order.
    unique_articles = []
    seen = set()

    for article in articles:
        number = article["article_number"]

        if number not in seen:
            unique_articles.append(article)
            seen.add(number)

    return unique_articles


# --------------------------------------------------
# Build chunks
# --------------------------------------------------

def build_chunks(articles: list[dict]) -> list[dict]:
    """
    Convert article sections into searchable chunks.

    For this first version, each article is one chunk.
    Later we can split long articles into paragraph-level chunks.
    """

    chunks = []

    for article in articles:
        chunks.append(
            {
                "text": article["text"],
                "metadata": {
                    "article_number": article["article_number"],
                    "article_title": article["title"],
                    "category": "EU AI Act provision",
                    "source": "Regulation (EU) 2024/1689",
                    "document_version": DOCUMENT_VERSION,
                    "source_url": SOURCE_URL,
                },
            }
        )

    return chunks


# --------------------------------------------------
# Store in ChromaDB
# --------------------------------------------------

def index_chunks(chunks: list[dict]) -> None:
    """
    Rebuild the regulations collection from the official document.
    """

    embedding_function = (
        embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )

    client = chromadb.PersistentClient(path=str(DB_PATH))

    # Rebuild the collection so old demo provisions disappear.
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Removed previous regulations collection.")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )

    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    ids = [
        f"eu_ai_act_article_{chunk['metadata']['article_number']}"
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"Indexed {len(chunks)} regulatory articles.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Database path: {DB_PATH}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print("Reading official EU AI Act HTML...")

    text = load_html_text(HTML_PATH)

    print(f"Extracted {len(text):,} characters of text.")

    articles = extract_articles(text)

    print(f"Detected {len(articles)} articles.")

    chunks = build_chunks(articles)

    index_chunks(chunks)

    print("\nIngestion complete!")

    print("\nFirst five indexed articles:")
    for article in chunks[:5]:
        metadata = article["metadata"]
        print(
            f"- Article {metadata['article_number']}: "
            f"{metadata['article_title']}"
        )


if __name__ == "__main__":
    main()