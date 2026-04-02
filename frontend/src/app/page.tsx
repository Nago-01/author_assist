'use client';

import { useState, useRef, DragEvent, ChangeEvent } from 'react';
import ResultsDashboard from '@/components/ResultsDashboard';
import LoadingState from '@/components/LoadingState';

// ── Types ─────────────────────────────────────────────────────────────────────

export type AnalysisResult = {
  title: {
    final_title: string;
    rationale: string;
    alternative_titles: string[];
    all_candidates: string[];
  };
  tldr: {
    final_tldr: string;
    one_liner: string;
    key_points: string[];
  };
  tags: {
    final_tags: Array<{ tag: string; category: string; rationale: string }>;
    candidate_counts: { gazetteer: number; spacy: number; llm: number; total_deduped: number };
  };
  references: {
    final_references: Array<{ formatted?: string; raw?: string }>;
    citation_style: string;
    total_references: number;
  };
  meta: {
    revision_rounds: number;
    review_verdicts: Record<string, string>;
    shared_context: {
      key_themes: string[];
      target_audience: string;
      main_message: string;
      domain: string;
      article_type: string;
    };
    timestamp: string;
  };
};

// ── Constants ─────────────────────────────────────────────────────────────────

// In production, set this to your Hugging Face Space URL
const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

// ── Main Page ─────────────────────────────────────────────────────────────────

export default function Home() {
  const [mode, setMode] = useState<'file' | 'text'>('file');
  const [file, setFile] = useState<File | null>(null);
  const [rawText, setRawText] = useState('');
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // ── Drag & drop handlers ───────────────────────────────────────────────────

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) setFile(dropped);
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const picked = e.target.files?.[0];
    if (picked) setFile(picked);
  };

  // ── Submit ─────────────────────────────────────────────────────────────────

  const handleAnalyze = async () => {
    setError(null);
    setResult(null);
    setLoading(true);

    try {
      let response: Response;

      if (mode === 'file' && file) {
        const formData = new FormData();
        formData.append('file', file);
        response = await fetch(`${API_BASE}/analyze/file`, {
          method: 'POST',
          body: formData,
        });
      } else {
        response = await fetch(`${API_BASE}/analyze/text`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: rawText }),
        });
      }

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data?.detail ?? `Server error (${response.status})`);
      }

      const data: AnalysisResult = await response.json();
      setResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Unexpected error');
    } finally {
      setLoading(false);
    }
  };

  const canSubmit =
    !loading && (mode === 'file' ? file !== null : rawText.trim().length > 50);

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <main className="page-wrapper">

      {/* Hero */}
      <header className="hero">
        <div className="hero-logo">Author Assist</div>
        <p className="hero-tagline">
          AI-powered metadata generation for publications. Upload a paper and get
          titles, TLDRs, tags, and references in seconds.
        </p>
      </header>

      {/* Input section */}
      {!result && !loading && (
        <section>
          {/* Mode toggle */}
          <div className="mode-toggle">
            <button
              id="mode-file"
              className={`mode-btn ${mode === 'file' ? 'active' : ''}`}
              onClick={() => setMode('file')}
            >
              Upload File
            </button>
            <button
              id="mode-text"
              className={`mode-btn ${mode === 'text' ? 'active' : ''}`}
              onClick={() => setMode('text')}
            >
              Paste Text
            </button>
          </div>

          {mode === 'file' ? (
            <div
              id="upload-zone"
              role="button"
              tabIndex={0}
              aria-label="File upload zone"
              className={`upload-zone ${dragging ? 'dragover' : ''}`}
              onClick={() => inputRef.current?.click()}
              onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
            >
              <div className="upload-icon">📂</div>
              <div className="upload-title">Drop your article here or click to browse</div>
              <div className="upload-sub">We&apos;ll extract the text automatically</div>
              <div className="upload-types">
                {['PDF', 'DOCX', 'DOC', 'TXT'].map((t) => (
                  <span key={t} className="type-pill">{t}</span>
                ))}
              </div>
              {file && (
                <div className="upload-file-chosen">✓ {file.name}</div>
              )}
              <input
                ref={inputRef}
                id="file-input"
                type="file"
                accept=".pdf,.docx,.doc,.txt"
                className="upload-hidden"
                onChange={handleFileChange}
              />
            </div>
          ) : (
            <textarea
              id="text-input"
              className="text-area"
              placeholder="Paste your article text here (minimum 50 characters)…"
              value={rawText}
              onChange={(e) => setRawText(e.target.value)}
            />
          )}

          <button
            id="analyze-btn"
            className="analyze-btn"
            disabled={!canSubmit}
            onClick={handleAnalyze}
          >
            Analyse Article →
          </button>

          {error && <div className="error-box">⚠ {error}</div>}
        </section>
      )}

      {/* Loading */}
      {loading && <LoadingState />}

      {/* Results */}
      {result && !loading && (
        <>
          <ResultsDashboard result={result} />
          <button
            id="reset-btn"
            className="reset-btn"
            onClick={() => {
              setResult(null);
              setFile(null);
              setRawText('');
              setError(null);
            }}
          >
            ← Analyse another article
          </button>
        </>
      )}
    </main>
  );
}
