import os

from dotenv import load_dotenv
from google import genai

from services.paper_search import search_papers
from services.embedding_service import create_embeddings
from services.vector_store import search_chunks


load_dotenv()


api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is not set in the .env file."
    )


client = genai.Client(
    api_key=api_key
)


# Cache successful paper searches
paper_cache = {}


def run_research_agent(query: str):

    try:

        print("\n==============================")
        print("RESEARCH AGENT STARTED")
        print("==============================")


        # ==========================================
        # STEP 1: SEARCH ACADEMIC PAPERS
        # ==========================================

        print("\nSearching academic papers...")


        papers = []


        if query in paper_cache:

            print("Using cached paper results.")

            papers = paper_cache[query]["papers"]

        else:

            paper_result = search_papers(
                query,
                limit=5
            )


            if paper_result.get("success"):

                papers = paper_result.get(
                    "papers",
                    []
                )

                paper_cache[query] = paper_result

                print(
                    f"Found {len(papers)} academic papers."
                )

            else:

                # IMPORTANT:
                # Do not stop the entire agent
                # if the external paper APIs fail.

                print(
                    "Live paper search unavailable."
                )

                print(
                    "Continuing with local RAG knowledge..."
                )


        # ==========================================
        # STEP 2: CREATE QUERY EMBEDDING
        # ==========================================

        print(
            "Creating question embedding..."
        )


        query_embedding = create_embeddings(
            [query]
        )[0]


        # ==========================================
        # STEP 3: SEARCH CHROMADB
        # ==========================================

        print(
            "Searching ChromaDB..."
        )


        retrieved = search_chunks(
            query_embedding,
            top_k=5
        )


        documents = retrieved.get(
            "documents",
            [[]]
        )[0]


        print(
            f"Retrieved {len(documents)} research chunks."
        )


        # ==========================================
        # STEP 4: BUILD PAPER CONTEXT
        # ==========================================

        if papers:

            paper_context_parts = []


            for index, paper in enumerate(
                papers,
                start=1
            ):

                title = paper.get(
                    "title",
                    "Unknown title"
                )


                authors = ", ".join(
                    paper.get(
                        "authors",
                        []
                    )
                )


                year = paper.get(
                    "year",
                    "Unknown year"
                )


                url = paper.get(
                    "url",
                    "N/A"
                )


                paper_context_parts.append(
                    f"""
Paper {index}

Title: {title}

Authors: {authors}

Year: {year}

URL: {url}
"""
                )


            paper_context = "\n".join(
                paper_context_parts
            )

        else:

            paper_context = """
No live academic paper metadata is
currently available because the external
paper-search APIs are temporarily
rate-limited.

Use the retrieved research evidence
from the local research database.
"""


        # ==========================================
        # STEP 5: BUILD RESEARCH CONTEXT
        # ==========================================

        research_context = "\n\n".join(
            documents
        )


        # ==========================================
        # STEP 6: CREATE GEMINI PROMPT
        # ==========================================

        prompt = f"""
You are an AI academic research assistant.

Research question:

{query}


ACADEMIC PAPER SEARCH RESULTS:

{paper_context}


RETRIEVED RESEARCH EVIDENCE:

{research_context}


Instructions:

1. Answer the research question clearly.

2. Use the retrieved research evidence
   as the main evidence for your answer.

3. Do not invent information.

4. Explain the important techniques and
   concepts in a structured way.

5. Mention applications and challenges
   when supported by the evidence.

6. If the evidence is insufficient for
   a claim, clearly say so.

7. Keep the answer concise but informative.

8. If paper metadata is available, finish
   with a section called:

   Research Sources

   and list the relevant paper titles
   and URLs.

9. If live paper metadata is unavailable,
   do not invent paper titles or URLs.
"""


        # ==========================================
        # STEP 7: CALL GEMINI
        # ==========================================

        print(
            "Sending request to Gemini..."
        )


        interaction = client.interactions.create(
            model="gemini-3.8-flash",
            input=prompt,
            generation_config={
                "thinking_level": "low"
            }
        )


        answer = interaction.output_text


        print(
            "Gemini response received."
        )


        # ==========================================
        # STEP 8: RETURN RESULT
        # ==========================================

        return {

            "success": True,

            "query": query,

            "answer": answer,

            "papers": papers,

            "retrieved_chunks": documents

        }


    except Exception as e:

        print(
            "\nRESEARCH AGENT ERROR:"
        )

        print(
            repr(e)
        )


        return {

            "success": False,

            "error": (
                "The AI research agent encountered "
                f"an error: {str(e)}"
            )

        }