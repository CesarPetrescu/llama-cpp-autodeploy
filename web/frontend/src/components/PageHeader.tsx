import { ReactNode } from "react";

interface Props {
  eyebrow?: ReactNode;
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
}

export function PageHeader({ eyebrow, title, description, actions }: Props) {
  return (
    <header className="brand-surface mb-6 flex flex-col gap-4 px-4 py-4 sm:gap-5 sm:px-6 sm:py-5 sm:flex-row sm:items-start sm:justify-between">
      <div className="min-w-0 sm:flex-1">
        {eyebrow && (
          <div className="brand-chip mb-3">
            <span className="h-1.5 w-1.5 rounded-none bg-lime-300" />
            {eyebrow}
          </div>
        )}
        <h1 className="font-display text-[1.85rem] font-bold tracking-tight text-bone-50 sm:text-3xl md:text-[2.75rem]">
          {title}
        </h1>
        {description && (
          <p className="mt-2 max-w-3xl text-sm leading-6 text-bone-300 md:mt-3 md:text-[15px]">
            {description}
          </p>
        )}
      </div>
      {actions && (
        <div className="flex min-w-0 w-full flex-wrap items-stretch gap-2 sm:w-auto sm:max-w-[40rem] sm:items-start sm:justify-end">
          {actions}
        </div>
      )}
    </header>
  );
}
