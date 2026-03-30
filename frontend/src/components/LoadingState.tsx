export default function LoadingState() {
  const steps = [
    { id: 'manager',    label: 'Manager',    icon: '🧠' },
    { id: 'agents',     label: 'Agents',     icon: '⚡' },
    { id: 'reviewer',   label: 'Reviewer',   icon: '🔍' },
  ];

  return (
    <div className="loading-wrap">
      <div className="spinner" />
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', marginBottom: 4 }}>
        Analysing your article…
      </p>
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginBottom: 24 }}>
        This usually takes 20–60 seconds depending on article length.
      </p>

      <div className="steps-row">
        {steps.map((s) => (
          <div key={s.id} className="step-badge active">
            <span className="pulse" />
            {s.icon} {s.label}
          </div>
        ))}
      </div>

      <p style={{ color: 'var(--text-secondary)', fontSize: '0.78rem', marginTop: 8 }}>
        Four agents running in parallel · Reviewer gives targeted feedback
      </p>
    </div>
  );
}
