import { useState } from "react";

function App() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [papers, setPapers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const runResearch = async () => {
    if (!query.trim()) {
      setError("Please enter a research question.");
      return;
    }

    setLoading(true);
    setError("");
    setAnswer("");
    setPapers([]);

    try {
      // IMPORTANT:
      // We use /api instead of directly calling port 8000.
      // Vite will proxy this request to FastAPI.
      const url = `/api/research-agent?query=${encodeURIComponent(
        query
      )}`;

      console.log("Sending request to:", url);

      const response = await fetch(url);

      console.log("Backend response status:", response.status);

      const responseText = await response.text();

      console.log("Backend response:", responseText);

      if (!response.ok) {
        throw new Error(
          `Backend returned HTTP ${response.status}`
        );
      }

      let data;

      try {
        data = JSON.parse(responseText);
      } catch {
        throw new Error("Backend returned invalid JSON.");
      }

      console.log("Research result:", data);

      if (!data.success) {
        throw new Error(
          data.error || "Research agent failed."
        );
      }

      setAnswer(data.answer || "");
      setPapers(data.papers || []);
    } catch (err) {
      console.error("Research request error:", err);

      setError(
        err.message ||
          "Unable to connect to the research agent."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header>
        <h1>🔬 AI Academic Research Agent</h1>

        <p>
          Search papers, retrieve research evidence,
          and generate AI-powered research answers.
        </p>
      </header>

      <main>
        {/* SEARCH */}
        <div className="search-box">
          <input
            type="text"
            placeholder="Ask a research question..."
            value={query}
            onChange={(event) =>
              setQuery(event.target.value)
            }
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                runResearch();
              }
            }}
          />

          <button
            onClick={runResearch}
            disabled={loading}
          >
            {loading
              ? "🔄 Researching..."
              : "🔍 Start Research"}
          </button>
        </div>

        {/* ERROR */}
        {error && (
          <div className="error">
            ❌ {error}
          </div>
        )}

        {/* LOADING */}
        {loading && (
          <div className="loading">
            <div className="spinner"></div>

            <p>
              Searching academic papers, retrieving
              research evidence, and generating your
              answer...
            </p>
          </div>
        )}

        {/* AI ANSWER */}
        {answer && !loading && (
          <section className="answer-section">
            <h2>🤖 AI Research Answer</h2>

            <div className="answer-card">
              <div className="answer-text">
                {answer}
              </div>
            </div>
          </section>
        )}

        {/* PAPERS */}
        {papers.length > 0 && !loading && (
          <section>
            <h2>📚 Related Academic Papers</h2>

            <div className="papers">
              {papers.map((paper, index) => (
                <article
                  className="paper-card"
                  key={
                    paper.url ||
                    paper.title ||
                    index
                  }
                >
                  <h3>{paper.title}</h3>

                  <p className="authors">
                    {paper.authors?.length
                      ? paper.authors.join(", ")
                      : "Unknown authors"}

                    {paper.year
                      ? ` • ${paper.year}`
                      : ""}
                  </p>

                  {paper.abstract && (
                    <p className="abstract">
                      {paper.abstract}
                    </p>
                  )}

                  {paper.url && (
                    <a
                      href={paper.url}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Read Paper ↗
                    </a>
                  )}
                </article>
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;