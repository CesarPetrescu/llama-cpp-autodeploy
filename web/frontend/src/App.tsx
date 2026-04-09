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
  badge?: string;
  badgeClass?: string;
}

export default function App() {
  usePrefetch();
  const feeds = useLiveFeeds();

  const nav: NavItem[] = [
    { to: "/dashboard", label: "Dashboard" },
    {
      to: "/instances",
      label: "Instances",
      badge: `${feeds.instances.running}/${feeds.instances.total}`,
      badgeClass:
        feeds.instances.running > 0
          ? "bg-emerald-500/20 text-emerald-300"
          : "bg-slate-700 text-slate-300",
    },
    { to: "/memory", label: "Memory" },
    { to: "/library", label: "Library" },
    {
      to: "/builds",
      label: "Builds",
      badge:
        feeds.builds.running > 0 ? `${feeds.builds.running} running` : undefined,
      badgeClass: "bg-amber-500/20 text-amber-300",
    },
    { to: "/settings", label: "Settings" },
  ];

  return (
    <div className="flex min-h-screen">
      <aside className="w-56 shrink-0 border-r border-slate-800 bg-slate-900/60 p-4">
        <h1 className="mb-2 text-lg font-semibold tracking-tight">
          llama-cpp
          <span className="block text-xs font-normal text-slate-400">
            autodeploy
          </span>
        </h1>
        <p className="mb-5 text-xs text-slate-500">
          live: {feeds.connected ? (
            <span className="text-emerald-400">●</span>
          ) : (
            <span className="text-rose-400">●</span>
          )}
        </p>
        <nav className="flex flex-col gap-1">
          {nav.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                [
                  "flex items-center justify-between gap-2 rounded px-3 py-2 text-sm transition",
                  isActive
                    ? "bg-sky-500/20 text-sky-200"
                    : "text-slate-300 hover:bg-slate-800",
                ].join(" ")
              }
            >
              <span>{item.label}</span>
              {item.badge && (
                <span
                  className={`rounded-full px-2 py-0.5 text-[10px] font-medium tracking-wide ${
                    item.badgeClass ?? "bg-slate-700 text-slate-300"
                  }`}
                >
                  {item.badge}
                </span>
              )}
            </NavLink>
          ))}
        </nav>
      </aside>

      <main className="flex-1 overflow-x-hidden p-6">
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
      </main>
    </div>
  );
}
