'use client';

import { useState } from 'react';
import type { AnalysisResult } from '@/app/page';

function TagPill({ tag, category, rationale }: { tag: string; category: string; rationale: string }) {
  return (
    <span className="tag-pill" data-cat={category}>
      {tag}
      <span className="tag-tooltip">{rationale}</span>
    </span>
  );
}

function TitleCard({ data }: { data: AnalysisResult['title'] }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="glass-card result-card">
      <div className="result-card-header">
        <span className="result-card-icon">📌</span>
        <span className="result-card-title">Generated Title</span>
      </div>
      <p className="final-title">{data.final_title || 'N/A'}</p>
      {data.alternative_titles?.length > 0 && (
        <>
          <button
            id="alts-toggle-btn"
            className="alts-toggle"
            onClick={() => setOpen((o) => !o)}
          >
            {open ? '▴ Hide' : '▾ Show'} alternatives ({data.alternative_titles.length})
          </button>
          {open && (
            <ul className="alt-title-list">
              {data.alternative_titles.map((t, i) => (
                <li key={i}>{t}</li>
              ))}
            </ul>
          )}
        </>
      )}
    </div>
  );
}

function TagsCard({ data }: { data: AnalysisResult['tags'] }) {
  const tags = data.final_tags ?? [];
  return (
    <div className="glass-card result-card">
      <div className="result-card-header">
        <span className="result-card-icon">🏷️</span>
        <span className="result-card-title">Tags ({tags.length})</span>
      </div>
      <div className="tags-wrap">
        {tags.length === 0
          ? <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>No tags generated.</span>
          : tags.map((t, i) => (
              <TagPill key={i} tag={t.tag} category={t.category} rationale={t.rationale} />
            ))
        }
      </div>
    </div>
  );
}

function TLDRCard({ data }: { data: AnalysisResult['tldr'] }) {
  return (
    <div className="glass-card result-card">
      <div className="result-card-header">
        <span className="result-card-icon">📄</span>
        <span className="result-card-title">TLDR</span>
      </div>
      {data.one_liner && <p className="one-liner">&ldquo;{data.one_liner}&rdquo;</p>}
      <p className="tldr-text">{data.final_tldr || 'N/A'}</p>
    </div>
  );
}

function RefsCard({ data }: { data: AnalysisResult['references'] }) {
  const refs = data.final_references ?? [];
  return (
    <div className="glass-card result-card">
      <div className="result-card-header">
        <span className="result-card-icon">📚</span>
        <span className="result-card-title">
          References ({refs.length} · {data.citation_style ?? 'Unknown style'})
        </span>
      </div>
      <ul className="refs-list">
        {refs.slice(0, 8).map((r, i) => (
          <li key={i}>
            <span className="ref-num">{i + 1}</span>
            <span>{r.formatted ?? r.raw ?? JSON.stringify(r)}</span>
          </li>
        ))}
        {refs.length > 8 && (
          <li style={{ color: 'var(--text-secondary)', fontSize: '0.78rem', paddingTop: 8 }}>
            …and {refs.length - 8} more in the downloaded JSON
          </li>
        )}
      </ul>
    </div>
  );
}

export default function ResultsDashboard({ result }: { result: AnalysisResult }) {
  const { meta } = result;

  return (
    <>
      <div className="results-grid">
        <TitleCard data={result.title} />
        <TagsCard  data={result.tags} />
        <TLDRCard  data={result.tldr} />
        <RefsCard  data={result.references} />
      </div>

      <div className="meta-bar">
        <span>
          Revision rounds: <strong>{meta.revision_rounds}</strong> &nbsp;·&nbsp;
          Domain: <strong>{meta.shared_context?.domain ?? '—'}</strong> &nbsp;·&nbsp;
          {new Date(meta.timestamp).toLocaleTimeString()}
        </span>
        <div className="verdict-row">
          {Object.entries(meta.review_verdicts ?? {}).map(([agent, verdict]) => (
            <span
              key={agent}
              className={`verdict-chip ${verdict === 'approved' ? '' : 'revised'}`}
            >
              {verdict === 'approved' ? '✓' : '↺'} {agent.replace('_generator', '')}
            </span>
          ))}
        </div>
      </div>
    </>
  );
}
