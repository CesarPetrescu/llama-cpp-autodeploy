import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import Instances from "./pages/Instances";
import InstanceLogs from "./pages/InstanceLogs";
import Library from "./pages/Library";
import Memory from "./pages/Memory";
import Builds from "./pages/Builds";
import Settings from "./pages/Settings";
import { usePrefetch } from "./hooks/usePrefetch";
import { useLiveFeeds } from "./hooks/useLiveFeeds";

interface NavItem {
  to: string;
  label: string;
  icon: string;
  badge?: string;
  badgeClass?: string;
}

export default function App() {
  usePrefetch();
  const feeds = useLiveFeeds();

  const nav: NavItem[] = [
    { to: "/dashboard", label: "Overview", icon: "◆" },
    {
      to: "/instances",
      label: "Instances",
      icon: "◉",
      badge: `${feeds.instances.running}/${feeds.instances.total}`,
      badgeClass:
        feeds.instances.running > 0
          ? "bg-lime-300 text-ink-900"
          : "bg-white/10 text-bone-300",
    },
    { to: "/memory", label: "Memory", icon: "▤" },
    { to: "/library", label: "Library", icon: "▣" },
    {
      to: "/builds",
      label: "Builds",
      icon: "▲",
      badge:
        feeds.builds.running > 0 ? `${feeds.builds.running}` : undefined,
      badgeClass: "bg-amber-400/80 text-ink-900",
    },
    { to: "/settings", label: "Settings", icon: "✦" },
  ];

  return (
    <div className="relative flex min-h-screen text-bone-100">
      {/* Ambient background grid */}
      <div
        aria-hidden
        className="pointer-events-none fixed inset-0 opacity-[0.35] mix-blend-screen"
        style={{
          backgroundImage:
            "radial-gradient(rgba(255,255,255,0.035) 1px, transparent 1px)",
          backgroundSize: "22px 22px",
        }}
      />

      <aside className="relative z-10 flex w-64 shrink-0 flex-col gap-6 border-r border-white/5 bg-ink-500/80 px-5 py-6 backdrop-blur">
        {/* Brand lockup */}
        <div className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-lime-300 font-display text-xl font-black text-ink-900 shadow-glow-lime">
            λ
          </div>
          <div>
            <div className="font-display text-[15px] font-bold leading-tight tracking-tight text-bone-50">
              llama-cpp
            </div>
            <div className="text-[11px] uppercase tracking-[0.2em] text-bone-400">
              autodeploy
            </div>
          </div>
        </div>

        {/* Live indicator */}
        <div className="brand-surface-muted flex items-center justify-between px-3 py-2">
          <span className="text-[10px] font-semibold uppercase tracking-[0.2em] text-bone-400">
            live feed
          </span>
          <span className="flex items-center gap-1.5 text-[11px] font-semibold">
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                feeds.connected
                  ? "bg-lime-300 shadow-[0_0_12px_rgba(213,255,64,0.8)]"
                  : "bg-rose-400"
              }`}
            />
            <span
              className={feeds.connected ? "text-lime-200" : "text-rose-200"}
            >
              {feeds.connected ? "online" : "offline"}
            </span>
          </span>
        </div>

        <nav className="flex flex-col gap-1">
          {nav.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                [
                  "group relative flex items-center justify-between gap-2 rounded-xl px-3 py-2.5 text-sm font-medium transition",
                  isActive
                    ? "bg-lime-300/10 text-lime-200 ring-1 ring-lime-300/40"
                    : "text-bone-300 hover:bg-white/5 hover:text-bone-50",
                ].join(" ")
              }
            >
              {({ isActive }) => (
                <>
                  <span className="flex items-center gap-3">
                    <span
                      className={`flex h-7 w-7 items-center justify-center rounded-lg text-[13px] ${
                        isActive
                          ? "bg-lime-300 text-ink-900"
                          : "bg-white/5 text-bone-300 group-hover:text-lime-200"
                      }`}
                    >
                      {item.icon}
                    </span>
                    <span className="tracking-tight">{item.label}</span>
                  </span>
                  {item.badge && (
                    <span
                      className={`rounded-full px-2 py-0.5 text-[10px] font-bold tracking-wide ${
                        item.badgeClass ?? "bg-white/10 text-bone-300"
                      }`}
                    >
                      {item.badge}
                    </span>
                  )}
                </>
              )}
            </NavLink>
          ))}
        </nav>

      </aside>

      <main className="relative z-10 flex-1 overflow-x-hidden">
        <div className="mx-auto w-full max-w-6xl px-8 py-8">
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
  );
}
