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
    <section
      className={`brand-surface ${padded ? "p-5" : ""} ${className}`}
    >
      {(title || subtitle || actions) && (
        <header className="mb-4 flex items-start justify-between gap-4">
          <div className="flex flex-col gap-1">
            {title && <h2 className="brand-label">{title}</h2>}
            {subtitle && <p className="text-sm text-bone-300">{subtitle}</p>}
          </div>
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  );
}
