'use client';

import type { HistoryEntry } from '@/lib/history';

type Props = {
  entries: HistoryEntry[];
  onRestore: (entry: HistoryEntry) => void;
  onClear: () => void;
};

export default function HistoryPanel({ entries, onRestore, onClear }: Props) {
  if (entries.length === 0) return null;

  return (
    <section className="history-section">
      <div className="history-header">
        <span className="history-title">⏱ Recent Analyses</span>
        <button id="history-clear-btn" className="history-clear-btn" onClick={onClear}>
          Clear history
        </button>
      </div>

      <div className="history-list">
        {entries.map((entry, idx) => {
          const date = new Date(entry.timestamp);
          const dateStr = isNaN(date.getTime())
            ? 'Unknown date'
            : date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) +
              ' · ' +
              date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });

          const tagCount  = entry.result.tags?.final_tags?.length ?? 0;
          const refCount  = entry.result.references?.final_references?.length ?? 0;

          return (
            <button
              key={entry.id}
              id={`history-entry-${idx}`}
              className="history-card"
              onClick={() => onRestore(entry)}
              title="Click to view this result"
            >
              <div className="history-card-top">
                <span className="history-index">#{entries.length - idx}</span>
                <span className="history-date">{dateStr}</span>
              </div>
              <p className="history-label">{entry.label}</p>
              <div className="history-chips">
                <span className="history-chip">{tagCount} tags</span>
                <span className="history-chip">{refCount} refs</span>
                {entry.result.meta?.shared_context?.domain && (
                  <span className="history-chip">
                    {entry.result.meta.shared_context.domain}
                  </span>
                )}
              </div>
            </button>
          );
        })}
      </div>
    </section>
  );
}
