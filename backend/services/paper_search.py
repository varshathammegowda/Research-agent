import os
import time

import requests
from dotenv import load_dotenv


load_dotenv()


OPENALEX_URL = "https://api.openalex.org/works"

OPENALEX_API_KEY = os.getenv(
    "OPENALEX_API_KEY"
)


def reconstruct_abstract(inverted_index):
    """
    OpenAlex stores abstracts as an inverted index.
    This function reconstructs the original abstract.
    """

    if not inverted_index:
        return None

    words = []

    for word, positions in inverted_index.items():

        for position in positions:

            words.append(
                (position, word)
            )

    words.sort(
        key=lambda x: x[0]
    )

    abstract = " ".join(
        word
        for position, word in words
    )

    return abstract


def search_openalex(
    query: str,
    limit: int = 5,
    max_retries: int = 3
):
    """
    Search academic papers using OpenAlex.
    """

    # Remove question mark from the end
    # because OpenAlex can interpret ? as
    # a wildcard character.
    clean_query = query.strip().rstrip("?")

    params = {
        "search": clean_query,
        "per-page": limit
    }

    # Add API key if available
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY

    for attempt in range(max_retries):

        try:

            print(
                f"OpenAlex search "
                f"(attempt {attempt + 1})..."
            )

            response = requests.get(
                OPENALEX_URL,
                params=params,
                timeout=20
            )

            print(
                "OpenAlex Status:",
                response.status_code
            )

            # ==================================
            # SUCCESS
            # ==================================

            if response.status_code == 200:

                data = response.json()

                papers = []

                for work in data.get(
                    "results",
                    []
                ):

                    title = work.get(
                        "title"
                    )

                    if not title:
                        continue

                    title_lower = title.lower()

                    # Ignore erratum/correction papers
                    if "erratum" in title_lower:
                        continue

                    if "correction" in title_lower:
                        continue

                    # --------------------------
                    # AUTHORS
                    # --------------------------

                    authors = []

                    for author in work.get(
                        "authorships",
                        []
                    ):

                        author_info = author.get(
                            "author"
                        )

                        if author_info:

                            author_name = (
                                author_info.get(
                                    "display_name"
                                )
                            )

                            if author_name:
                                authors.append(
                                    author_name
                                )

                    # --------------------------
                    # ABSTRACT
                    # --------------------------

                    abstract = reconstruct_abstract(
                        work.get(
                            "abstract_inverted_index"
                        )
                    )

                    # --------------------------
                    # URL
                    # --------------------------

                    url = (
                        work.get("doi")
                        or work.get("id")
                    )

                    papers.append({

                        "title": title,

                        "authors": authors,

                        "year": work.get(
                            "publication_year"
                        ),

                        "abstract": abstract,

                        "url": url,

                        "source": "OpenAlex"

                    })

                print(
                    f"OpenAlex returned "
                    f"{len(papers)} papers."
                )

                return {

                    "success": True,

                    "source": "OpenAlex",

                    "count": len(papers),

                    "papers": papers

                }

            # ==================================
            # RATE LIMIT
            # ==================================

            elif response.status_code == 429:

                retry_after = (
                    response.headers.get(
                        "Retry-After"
                    )
                )

                if retry_after:

                    try:
                        wait_time = int(
                            retry_after
                        )

                    except ValueError:
                        wait_time = 5

                else:

                    wait_time = 2 ** attempt

                print(
                    f"OpenAlex rate limited. "
                    f"Waiting {wait_time} seconds..."
                )

                time.sleep(
                    wait_time
                )

            # ==================================
            # OTHER ERROR
            # ==================================

            else:

                print(
                    "OpenAlex failed:",
                    response.text
                )

                break

        except requests.exceptions.Timeout:

            print(
                "OpenAlex request timed out."
            )

            if attempt < max_retries - 1:

                wait_time = 2 ** attempt

                print(
                    f"Retrying in "
                    f"{wait_time} seconds..."
                )

                time.sleep(
                    wait_time
                )

        except requests.exceptions.RequestException as e:

            print(
                "OpenAlex request error:",
                e
            )

            break

    return {

        "success": False,

        "error": (
            "OpenAlex is temporarily "
            "unavailable or rate-limited."
        )

    }


def search_papers(
    query: str,
    limit: int = 5
):
    """
    Main academic paper search function.
    """

    print(
        "\nSearching OpenAlex..."
    )

    result = search_openalex(
        query=query,
        limit=limit
    )

    if result.get("success"):

        print(
            "OpenAlex search successful."
        )

        return result

    print(
        "OpenAlex search failed."
    )

    return {

        "success": False,

        "error": (
            "Unable to retrieve academic "
            "papers from OpenAlex."
        )

    }