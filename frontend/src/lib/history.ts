'use client';

import type { AnalysisResult } from '@/app/page';

export type HistoryEntry = {
  id: string;
  timestamp: string;
  label: string;   // final_title or first 60 chars of one_liner
  result: AnalysisResult;
};

const STORAGE_KEY = 'author_assist_history';
const MAX_ENTRIES = 3;

export function loadHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as HistoryEntry[]) : [];
  } catch {
    return [];
  }
}

export function saveToHistory(result: AnalysisResult): HistoryEntry[] {
  const existing = loadHistory();
  const entry: HistoryEntry = {
    id: Date.now().toString(),
    timestamp: result.meta?.timestamp ?? new Date().toISOString(),
    label:
      result.title?.final_title?.slice(0, 72) ||
      result.tldr?.one_liner?.slice(0, 72) ||
      'Untitled analysis',
    result,
  };

  // Prepend newest, keep only MAX_ENTRIES
  const updated = [entry, ...existing].slice(0, MAX_ENTRIES);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
  return updated;
}

export function clearHistory(): void {
  localStorage.removeItem(STORAGE_KEY);
}
