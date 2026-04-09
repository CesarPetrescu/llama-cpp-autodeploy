import { ReactNode } from "react";

interface Props {
  eyebrow?: ReactNode;
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
}

export function PageHeader({ eyebrow, title, description, actions }: Props) {
  return (
    <header className="mb-8 flex items-end justify-between gap-6">
      <div>
        {eyebrow && (
          <div className="brand-label mb-2 flex items-center gap-2">
            <span className="inline-block h-px w-6 bg-lime-300" />
            {eyebrow}
          </div>
        )}
        <h1 className="font-display text-3xl font-bold tracking-tight text-bone-50 md:text-4xl">
          {title}
        </h1>
        {description && (
          <p className="mt-2 max-w-2xl text-sm text-bone-300">{description}</p>
        )}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </header>
  );
}
