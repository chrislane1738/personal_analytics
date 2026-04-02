"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BarChart3,
  GitCompare,
  Settings2,
  Play,
  Activity,
  TrendingUp,
  Database,
  Crosshair,
  DollarSign,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface NavItem {
  label: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
}

const navItems: NavItem[] = [
  { label: "Runs", href: "/", icon: BarChart3 },
  { label: "Compare", href: "/compare", icon: GitCompare },
  { label: "Strategies", href: "/strategies", icon: Settings2 },
  { label: "Launch", href: "/launch", icon: Play },
  { label: "Monitor", href: "/monitor", icon: Activity },
  { label: "Walk-Fwd", href: "/walk-forward", icon: TrendingUp },
  { label: "Data", href: "/data", icon: Database },
  { label: "Eval", href: "/eval", icon: Crosshair },
  { label: "Paper", href: "/paper", icon: DollarSign },
];

function NavLink({
  item,
  isActive,
  expanded,
}: {
  item: NavItem;
  isActive: boolean;
  expanded: boolean;
}) {
  return (
    <Link
      href={item.href}
      className={cn(
        "group relative flex items-center gap-3 rounded-md px-2 py-2 text-sm transition-colors",
        isActive
          ? "bg-[#18181b] text-[#fafafa]"
          : "text-[#71717a] hover:bg-[#18181b] hover:text-[#a1a1aa]"
      )}
    >
      {/* Active indicator bar */}
      {isActive && (
        <span className="absolute left-0 top-1/2 h-5 w-0.5 -translate-y-1/2 rounded-full bg-[#f97316]" />
      )}
      <item.icon className="h-4 w-4 shrink-0" />
      {expanded && <span className="truncate">{item.label}</span>}
    </Link>
  );
}

export function Sidebar() {
  const pathname = usePathname();
  const [expanded, setExpanded] = useState(false);

  return (
    <aside
      onMouseEnter={() => setExpanded(true)}
      onMouseLeave={() => setExpanded(false)}
      className={cn(
        "flex h-screen flex-col border-r border-[#1a1a1a] bg-[#09090b] transition-all duration-200 ease-in-out",
        expanded ? "w-48" : "w-14"
      )}
    >
      {/* Logo */}
      <div className="flex h-14 items-center gap-2 border-b border-[#1a1a1a] px-3">
        <span className="text-lg font-bold tracking-tight text-[#f97316] font-mono">
          TB
        </span>
        {expanded && (
          <span className="text-sm font-medium text-zinc-300 truncate">
            Trading Bot
          </span>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex flex-1 flex-col gap-1 px-2 py-3">
        {navItems.map((item) => {
          const isActive =
            item.href === "/"
              ? pathname === "/"
              : pathname.startsWith(item.href);

          if (!expanded) {
            return (
              <Tooltip key={item.href}>
                <TooltipTrigger
                  delay={0}
                  render={
                    <NavLink
                      item={item}
                      isActive={isActive}
                      expanded={false}
                    />
                  }
                />
                <TooltipContent side="right" sideOffset={8}>
                  {item.label}
                </TooltipContent>
              </Tooltip>
            );
          }

          return (
            <NavLink
              key={item.href}
              item={item}
              isActive={isActive}
              expanded
            />
          );
        })}
      </nav>

      {/* Version footer */}
      <div className="border-t border-[#1a1a1a] px-3 py-2">
        <span className="font-mono text-[10px] text-[#71717a]">
          {expanded ? "v0.1.0-alpha" : "v0.1"}
        </span>
      </div>
    </aside>
  );
}
