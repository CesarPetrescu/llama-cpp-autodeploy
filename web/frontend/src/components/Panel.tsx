import { ReactNode } from "react";

interface Props {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  padded?: boolean;
}

export function Panel({
  title,
  subtitle,
  actions,
  children,
  className = "",
  padded = true,
}: Props) {
  return (
    <section className={`brand-surface ${padded ? "p-5 sm:p-6" : ""} ${className}`}>
      {(title || subtitle || actions) && (
        <header className="mb-5 flex flex-col gap-3 border-b border-white/10 pb-4 sm:flex-row sm:items-start sm:justify-between">
          <div className="flex min-w-0 flex-col gap-1">
            {title && (
              <h2 className="text-base font-semibold tracking-tight text-bone-50">
                {title}
              </h2>
            )}
            {subtitle && <p className="text-sm leading-6 text-bone-300">{subtitle}</p>}
          </div>
          {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  );
}
