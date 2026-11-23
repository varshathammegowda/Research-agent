import { useState } from "react";
import "./App.css";

const prompts = [
  "What are the latest approaches to RAG?",
  "How is AI transforming cybersecurity?",
  "What are the challenges in large language models?",
  "Compare modern approaches to autonomous AI.",
];

function Icon({ name, size = 20, ...props }) {
  const paths = {
    search: <><circle cx="11" cy="11" r="7" /><path d="m20 20-4-4" /></>,
    arrow: <><path d="M7 17 17 7M7 7h10v10" /></>,
    spark: <><path d="m12 3 1.9 5.8L20 11l-6.1 2.2L12 19l-1.9-5.8L4 11l6.1-2.2L12 3Z" /><path d="m19 14 1.1 2.9L23 18l-2.9 1.1L19 22l-1.1-2.9L15 18l2.9-1.1L19 14Z" /></>,
    book: <><path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v17H6.5A2.5 2.5 0 0 1 4 17.5v-12Z" /><path d="M4 16a2 2 0 0 1 2-2h14M8 7h8M8 10h6" /></>,
    arrowRight: <><path d="M5 12h14M13 6l6 6-6 6" /></>,
    alert: <><path d="M10.3 3.9 2.6 17.2A2 2 0 0 0 4.3 20h15.4a2 2 0 0 0 1.7-2.8L13.7 3.9a2 2 0 0 0-3.4 0Z" /><path d="M12 9v4m0 4h.01" /></>,
  };
  return <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" {...props}>{paths[name]}</svg>;
}

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
      const url = `/api/research-agent?query=${encodeURIComponent(query)}`;
      const response = await fetch(url);
      const responseText = await response.text();

      if (!response.ok) throw new Error(`Backend returned HTTP ${response.status}`);

      let data;
      try {
        data = JSON.parse(responseText);
      } catch {
        throw new Error("Backend returned invalid JSON.");
      }

      if (!data.success) throw new Error(data.error || "Research agent failed.");

      setAnswer(data.answer || "");
      setPapers(data.papers || []);
    } catch (err) {
      console.error("Research request error:", err);
      setError(err.message || "Unable to connect to the research agent.");
    } finally {
      setLoading(false);
    }
  };

  const hasResults = Boolean(answer || papers.length);

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="AI Academic Research Agent home">
          <span className="brand-mark"><Icon name="spark" size={18} /></span>
          <span className="brand-name">AARA<span className="brand-divider">/</span><span>Research Agent</span></span>
        </a>
        <nav className="top-nav" aria-label="Main navigation">
          <a href="#about">About</a>
          <span className="nav-status"><span className="status-dot" /> Research workspace</span>
        </nav>
      </header>

      <main id="top" className={`main-content${hasResults ? " has-results" : ""}`}>
        <section className="hero" aria-labelledby="page-title">
          <div className="hero-kicker"><span className="kicker-rule" /> A clearer path to discovery</div>
          <h1 id="page-title">AI Academic<br className="desktop-break" /> <span>Research Agent</span></h1>
          <p className="hero-copy">Search, explore, and discover academic knowledge with ease.</p>
          <div className="hero-note"><Icon name="spark" size={15} /><span>Evidence-led answers, grounded in academic research</span></div>
        </section>

        <section className="search-area" aria-label="Research search">
          <div className={`search-panel${loading ? " is-loading" : ""}`}>
            <div className="input-row">
              <Icon name="search" size={23} className="search-icon" />
              <input
                type="text"
                aria-label="Research question"
                placeholder="Ask anything about academic research..."
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={(event) => { if (event.key === "Enter" && !loading) runResearch(); }}
              />
              <button className="research-button" onClick={runResearch} disabled={loading}>
                {loading ? <><span className="button-spinner" /> Researching</> : <>Research <Icon name="arrowRight" size={17} /></>}
              </button>
            </div>
            <div className="search-meta"><span><kbd>↵</kbd> to search</span><span className="secure-note"><span className="mini-dot" /> Academic search</span></div>
          </div>

          {!hasResults && !loading && (
            <div className="suggestions">
              <div className="suggestion-heading"><span>TRY ASKING</span><span className="suggestion-line" /></div>
              <div className="prompt-list">
                {prompts.map((prompt) => (
                  <button key={prompt} className="prompt-chip" onClick={() => { setQuery(prompt); setError(""); }}>
                    <span>{prompt}</span><Icon name="arrow" size={15} />
                  </button>
                ))}
              </div>
            </div>
          )}
        </section>

        {error && <div className="error-banner" role="alert"><Icon name="alert" size={19} /><span>{error}</span></div>}

        {loading && (
          <section className="loading-card" aria-live="polite">
            <div className="loading-emblem"><span className="loading-orbit orbit-one" /><span className="loading-orbit orbit-two" /><span className="loading-core"><Icon name="spark" size={20} /></span></div>
            <div className="loading-copy"><span className="eyebrow">RESEARCH IN PROGRESS</span><h2>Researching your question</h2><p>Gathering relevant academic sources and preparing your answer.</p></div>
            <div className="loading-indicator"><span /><span /><span /></div>
          </section>
        )}

        {answer && !loading && (
          <section className="results-section answer-section">
            <div className="section-heading"><div><span className="eyebrow">SYNTHESIZED RESPONSE</span><h2><span className="heading-icon"><Icon name="spark" size={19} /></span>Research answer</h2></div><span className="result-label"><span className="status-dot" /> Ready to explore</span></div>
            <article className="answer-card"><div className="answer-accent" /><div className="answer-text">{answer}</div></article>
          </section>
        )}

        {papers.length > 0 && !loading && (
          <section className="results-section papers-section">
            <div className="section-heading"><div><span className="eyebrow">THE LITERATURE</span><h2><span className="heading-icon"><Icon name="book" size={19} /></span>Related papers</h2></div><span className="paper-count">{papers.length} {papers.length === 1 ? "paper" : "papers"}</span></div>
            <div className="papers">
              {papers.map((paper, index) => (
                <article className="paper-card" key={paper.url || paper.title || index}>
                  <div className="paper-index">{String(index + 1).padStart(2, "0")}</div>
                  <div className="paper-content">
                    <div className="paper-meta">{paper.year && <><span>{paper.year}</span><span className="meta-separator">·</span></>}<span>{paper.authors?.length ? paper.authors.join(", ") : "Unknown authors"}</span></div>
                    <h3>{paper.title}</h3>
                    {paper.abstract && <p className="abstract">{paper.abstract}</p>}
                    {paper.url && <a className="paper-link" href={paper.url} target="_blank" rel="noopener noreferrer">Read paper <Icon name="arrow" size={15} /></a>}
                  </div>
                </article>
              ))}
            </div>
          </section>
        )}
      </main>

      <footer id="about" className="page-footer"><span className="footer-brand"><span className="footer-mark"><Icon name="spark" size={14} /></span>AI Academic Research Agent</span><span className="footer-caption">Research, made more accessible.</span></footer>
    </div>
  );
}

export default App;
