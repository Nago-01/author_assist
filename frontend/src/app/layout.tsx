import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Author Assist — AI Metadata Generator',
  description:
    'Multi-agent system that auto-generates titles, TLDRs, tags, and references from any academic article.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
