import { ReactNode } from "react";

interface Props {
  eyebrow?: ReactNode;
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
}

export function PageHeader({ eyebrow, title, description, actions }: Props) {
  return (
    <header className="brand-surface mb-6 flex flex-col gap-5 px-5 py-5 sm:flex-row sm:items-start sm:justify-between sm:px-6">
      <div className="min-w-0 sm:flex-1">
        {eyebrow && (
          <div className="brand-chip mb-3">
            <span className="h-1.5 w-1.5 rounded-none bg-lime-300" />
            {eyebrow}
          </div>
        )}
        <h1 className="font-display text-3xl font-bold tracking-tight text-bone-50 md:text-[2.75rem]">
          {title}
        </h1>
        {description && (
          <p className="mt-3 max-w-3xl text-sm leading-6 text-bone-300 md:text-[15px]">
            {description}
          </p>
        )}
      </div>
      {actions && (
        <div className="flex min-w-0 w-full flex-wrap items-start gap-2 sm:w-auto sm:max-w-[40rem] sm:justify-end">
          {actions}
        </div>
      )}
    </header>
  );
}
