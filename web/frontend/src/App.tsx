import { useEffect, useState } from "react";
import {
  NavLink,
  Navigate,
  Route,
  Routes,
  useLocation,
} from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import Instances from "./pages/Instances";
import InstanceLogs from "./pages/InstanceLogs";
import Library from "./pages/Library";
import Memory from "./pages/Memory";
import Builds from "./pages/Builds";
import Settings from "./pages/Settings";
import { usePrefetch } from "./hooks/usePrefetch";
import { useLiveFeeds } from "./hooks/useLiveFeeds";

type NavIconName =
  | "overview"
  | "instances"
  | "memory"
  | "library"
  | "builds"
  | "settings";

interface NavItem {
  to: string;
  label: string;
  description: string;
  icon: NavIconName;
  badge?: string;
  badgeClass?: string;
}

function NavIcon({ name, active }: { name: NavIconName; active?: boolean }) {
  const stroke = active ? "currentColor" : "rgba(219,228,239,0.92)";
  const common = {
    fill: "none",
    stroke,
    strokeWidth: 1.8,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
  };

  switch (name) {
    case "overview":
      return (
        <svg viewBox="0 0 24 24" className="h-4 w-4" aria-hidden>
          <path {...common} d="M4 12h6V5H4zM14 19h6v-9h-6zM14 5h6v4h-6zM4 19h6v-3H4z" />
        </svg>
      );
    case "instances":
      return (
        <svg viewBox="0 0 24 24" className="h-4 w-4" aria-hidden>
          <path {...common} d="M5 7h14M5 12h14M5 17h14" />
          <circle cx="8" cy="7" r="1.5" fill={stroke} />
          <circle cx="12" cy="12" r="1.5" fill={stroke} />
          <circle cx="16" cy="17" r="1.5" fill={stroke} />
        </svg>
      );
    case "memory":
      return (
        <svg viewBox="0 0 24 24" className="h-4 w-4" aria-hidden>
          <rect {...common} x="5" y="7" width="14" height="10" rx="2" />
          <path {...common} d="M9 4v3M15 4v3M9 17v3M15 17v3M3 10h2M3 14h2M19 10h2M19 14h2" />
        </svg>
      );
    case "library":
      return (
        <svg viewBox="0 0 24 24" className="h-4 w-4" aria-hidden>
          <path {...common} d="M5 6.5A2.5 2.5 0 0 1 7.5 4H19v15H7.5A2.5 2.5 0 0 0 5 21z" />
          <path {...common} d="M5 6.5V21M9 8h6" />
        </svg>
      );
    case "builds":
      return (
        <svg viewBox="0 0 24 24" className="h-4 w-4" aria-hidden>
          <path {...common} d="M14 6 8 12l6 6M10 6l6 6-6 6" />
        </svg>
      );
    case "settings":
      return (
        <svg viewBox="0 0 24 24" className="h-4 w-4" aria-hidden>
          <path
            {...common}
            d="M12 3v3M12 18v3M4.93 4.93l2.12 2.12M16.95 16.95l2.12 2.12M3 12h3M18 12h3M4.93 19.07l2.12-2.12M16.95 7.05l2.12-2.12"
          />
          <circle {...common} cx="12" cy="12" r="3.5" />
        </svg>
      );
  }

  return null;
}

function ShellStat({
  label,
  value,
  tone = "default",
}: {
  label: string;
  value: string;
  tone?: "default" | "positive" | "warning";
}) {
  const toneClass =
    tone === "positive"
      ? "text-lime-200"
      : tone === "warning"
        ? "text-amber-200"
        : "text-bone-50";
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.03] px-3 py-2.5">
      <div className="brand-label">{label}</div>
      <div className={`mt-2 text-xl font-semibold tracking-tight ${toneClass}`}>
        {value}
      </div>
    </div>
  );
}

export default function App() {
  usePrefetch();
  const location = useLocation();
  const feeds = useLiveFeeds();
  const [navOpen, setNavOpen] = useState(false);

  useEffect(() => {
    setNavOpen(false);
  }, [location.pathname]);

  const nav: NavItem[] = [
    {
      to: "/dashboard",
      label: "Overview",
      description: "health, GPUs, endpoints",
      icon: "overview",
    },
    {
      to: "/instances",
      label: "Instances",
      description: "launch and recover servers",
      icon: "instances",
      badge: `${feeds.instances.running}/${feeds.instances.total}`,
      badgeClass:
        feeds.instances.running > 0
          ? "bg-lime-300 text-ink-900"
          : "bg-white/10 text-bone-300",
    },
    {
      to: "/memory",
      label: "Memory",
      description: "placement and planning",
      icon: "memory",
    },
    {
      to: "/library",
      label: "Library",
      description: "GGUF inventory",
      icon: "library",
    },
    {
      to: "/builds",
      label: "Builds",
      description: "toolchain and logs",
      icon: "builds",
      badge: feeds.builds.running > 0 ? `${feeds.builds.running}` : undefined,
      badgeClass: "bg-amber-400/80 text-ink-900",
    },
    {
      to: "/settings",
      label: "Settings",
      description: "backend auth and network",
      icon: "settings",
    },
  ];

  const currentNav =
    nav.find(
      (item) =>
        location.pathname === item.to ||
        location.pathname.startsWith(`${item.to}/`),
    ) ?? nav[0];

  return (
    <div className="min-h-screen text-bone-100">
      <a href="#main-content" className="skip-link">
        Skip to main content
      </a>

      <div
        aria-hidden
        className="pointer-events-none fixed inset-0 opacity-60"
        style={{
          backgroundImage:
            "linear-gradient(rgba(148,163,184,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(148,163,184,0.05) 1px, transparent 1px)",
          backgroundSize: "48px 48px",
          maskImage:
            "radial-gradient(circle at center, black 35%, rgba(0,0,0,0.15) 80%, transparent 100%)",
        }}
      />

      <div className="relative min-h-screen lg:grid lg:grid-cols-[19rem_minmax(0,1fr)]">
        <button
          type="button"
          aria-label="Close navigation"
          onClick={() => setNavOpen(false)}
          className={`fixed inset-0 z-40 bg-ink-900/70 backdrop-blur-sm transition lg:hidden ${
            navOpen
              ? "pointer-events-auto opacity-100"
              : "pointer-events-none opacity-0"
          }`}
        />

        <aside
          className={`fixed inset-y-0 left-0 z-50 flex w-[18.5rem] flex-col border-r border-white/10 bg-ink-500/95 px-5 py-5 shadow-2xl backdrop-blur-xl transition-transform duration-200 lg:sticky lg:top-0 lg:h-screen lg:translate-x-0 ${
            navOpen ? "translate-x-0" : "-translate-x-full"
          }`}
        >
          <div className="flex items-start justify-between gap-3">
            <div className="flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-lime-300 text-lg font-black text-ink-900 shadow-glow-lime">
                λ
              </div>
              <div>
                <div className="font-display text-lg font-bold leading-tight text-bone-50">
                  llama-cpp
                </div>
                <div className="brand-label mt-1">autodeploy control plane</div>
              </div>
            </div>

            <button
              type="button"
              onClick={() => setNavOpen(false)}
              className="brand-btn-ghost px-3 py-2 lg:hidden"
            >
              Close
            </button>
          </div>

          <div className="mt-6 brand-surface p-4">
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="brand-label">Backend feed</div>
                <div className="mt-2 text-lg font-semibold text-bone-50">
                  {feeds.connected ? "Connected" : "Disconnected"}
                </div>
              </div>
              <span
                className={`h-3 w-3 rounded-full ${
                  feeds.connected
                    ? "bg-lime-300 shadow-[0_0_18px_rgba(213,255,64,0.9)]"
                    : "bg-rose-400 shadow-[0_0_18px_rgba(251,113,133,0.65)]"
                }`}
              />
            </div>
            <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-1">
              <ShellStat
                label="Running instances"
                value={`${feeds.instances.running}`}
                tone={feeds.instances.running > 0 ? "positive" : "default"}
              />
              <ShellStat
                label="Active builds"
                value={`${feeds.builds.running}`}
                tone={feeds.builds.running > 0 ? "warning" : "default"}
              />
            </div>
          </div>

          <nav className="mt-6 flex flex-1 flex-col gap-1" aria-label="Primary">
            {nav.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  [
                    "group rounded-2xl border px-3.5 py-3 text-left",
                    isActive
                      ? "border-lime-300/35 bg-lime-300/10 text-lime-100 shadow-[0_12px_30px_-22px_rgba(213,255,64,0.55)]"
                      : "border-transparent text-bone-300 hover:border-white/10 hover:bg-white/[0.04] hover:text-bone-50",
                  ].join(" ")
                }
              >
                {({ isActive }) => (
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex min-w-0 items-center gap-3">
                      <div
                        className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl border ${
                          isActive
                            ? "border-lime-300/50 bg-lime-300 text-ink-900"
                            : "border-white/10 bg-white/[0.04] text-bone-200"
                        }`}
                      >
                        <NavIcon name={item.icon} active={isActive} />
                      </div>
                      <div className="min-w-0">
                        <div className="text-sm font-semibold tracking-tight">
                          {item.label}
                        </div>
                        <div className="truncate text-xs text-bone-400">
                          {item.description}
                        </div>
                      </div>
                    </div>
                    {item.badge && (
                      <span
                        className={`shrink-0 rounded-full px-2 py-1 text-[10px] font-bold tracking-[0.16em] ${
                          item.badgeClass ?? "bg-white/10 text-bone-300"
                        }`}
                      >
                        {item.badge}
                      </span>
                    )}
                  </div>
                )}
              </NavLink>
            ))}
          </nav>

          <div className="brand-surface-muted mt-4 p-4">
            <div className="brand-label">Current section</div>
            <div className="mt-2 text-base font-semibold text-bone-50">
              {currentNav.label}
            </div>
            <p className="mt-1 text-sm text-bone-300">{currentNav.description}</p>
          </div>
        </aside>

        <div className="min-w-0">
          <header className="sticky top-0 z-30 border-b border-white/10 bg-ink-700/85 px-4 py-3 backdrop-blur lg:hidden">
            <div className="flex items-center justify-between gap-3">
              <div className="flex min-w-0 items-center gap-3">
                <button
                  type="button"
                  onClick={() => setNavOpen(true)}
                  className="brand-btn-ghost px-3 py-2"
                >
                  Menu
                </button>
                <div className="min-w-0">
                  <div className="brand-label">llama-cpp autodeploy</div>
                  <div className="truncate text-sm font-semibold text-bone-50">
                    {currentNav.label}
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5 text-xs font-semibold">
                <span
                  className={`h-2 w-2 rounded-full ${
                    feeds.connected ? "bg-lime-300" : "bg-rose-400"
                  }`}
                />
                {feeds.instances.running} live
              </div>
            </div>
          </header>

          <main
            id="main-content"
            className="relative px-4 py-4 sm:px-6 sm:py-6 lg:px-10 lg:py-8"
          >
            <div className="mx-auto w-full max-w-7xl">
              <Routes>
                <Route path="/" element={<Navigate to="/dashboard" replace />} />
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/instances" element={<Instances />} />
                <Route path="/instances/:id/logs" element={<InstanceLogs />} />
                <Route path="/memory" element={<Memory />} />
                <Route path="/library" element={<Library />} />
                <Route path="/builds" element={<Builds />} />
                <Route path="/settings" element={<Settings />} />
              </Routes>
            </div>
          </main>
        </div>
      </div>
    </div>
  );
}
