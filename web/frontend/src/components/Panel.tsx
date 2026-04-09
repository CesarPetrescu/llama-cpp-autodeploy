import { ReactNode } from "react";

interface Props {
  title?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}

export function Panel({ title, actions, children, className = "" }: Props) {
  return (
    <section
      className={`rounded-lg border border-slate-800 bg-slate-900/60 p-4 shadow ${className}`}
    >
      {(title || actions) && (
        <header className="mb-3 flex items-center justify-between">
          {title && <h2 className="text-sm font-semibold tracking-wide uppercase text-slate-300">{title}</h2>}
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  );
}
