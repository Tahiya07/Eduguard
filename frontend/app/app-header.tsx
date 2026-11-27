"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { clearSession, useSession } from "./session";

const links = [
  { href: "/", label: "Home" },
  { href: "/student", label: "Student" },
  { href: "/teacher", label: "Teacher" },
];

export default function AppHeader() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const session = useSession();
  const menuButton = useRef<HTMLButtonElement>(null);
  const panel = useRef<HTMLDivElement>(null);

  const active = (href: string) => (href === "/" ? pathname === href : pathname.startsWith(href));

  const closeMenu = useCallback(() => {
    setOpen(false);
    menuButton.current?.focus();
  }, []);

  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape" && open) closeMenu();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [open, closeMenu]);

  // The closed drawer must not be reachable with Tab.
  useEffect(() => {
    const el = panel.current;
    if (!el) return;
    if (open) el.removeAttribute("inert");
    else el.setAttribute("inert", "");
  }, [open]);

  const signOut = () => {
    clearSession();
    setOpen(false);
  };

  return (
    <>
      <header className="nav-pill-wrap sticky top-0 z-40 flex justify-center px-4 pt-5 sm:pt-7">
        <nav className="nav-pill" aria-label="Primary navigation">
          <Link href="/" onClick={() => setOpen(false)} className="nav-pill-brand">
            <img src="/framework-icon.svg" alt="" aria-hidden="true" className="h-6 w-6" />
            EduGuard
          </Link>
          <span className="nav-pill-divider" aria-hidden="true" />
          <div className="nav-pill-links">
            {links.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={`floating-nav-link ${active(link.href) ? "floating-nav-link-active" : ""}`}
                aria-current={active(link.href) ? "page" : undefined}
              >
                {link.label}
              </Link>
            ))}
            {session && (
              <button type="button" className="floating-nav-link nav-pill-signout" onClick={signOut}>
                Sign out
              </button>
            )}
          </div>
          <button
            type="button"
            ref={menuButton}
            className="nav-pill-menu"
            aria-expanded={open}
            aria-controls="mobile-navigation"
            aria-label="Open navigation menu"
            onClick={() => setOpen(true)}
          >
            <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round">
              <path d="M4 7h16M4 12h16M4 17h16" />
            </svg>
          </button>
        </nav>
      </header>

      <div ref={panel} id="mobile-navigation" className={`mobile-nav ${open ? "mobile-nav-open" : ""}`} aria-hidden={!open}>
        <button type="button" className="mobile-nav-backdrop" onClick={closeMenu} aria-label="Close navigation menu" tabIndex={open ? 0 : -1} />
        <aside className="mobile-nav-panel" role="dialog" aria-modal="true" aria-label="EduGuard navigation">
          <div className="mobile-nav-heading">
            <div>
              <span className="mobile-nav-kicker">EduGuard</span>
              <h2>Explore your workspace</h2>
            </div>
            <button type="button" onClick={closeMenu} className="mobile-nav-close" aria-label="Close navigation menu">
              <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <path d="m6 6 12 12M18 6 6 18" />
              </svg>
            </button>
          </div>
          <nav className="mobile-nav-links" aria-label="Mobile navigation">
            {links.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                onClick={() => setOpen(false)}
                aria-current={active(link.href) ? "page" : undefined}
                className={`mobile-nav-link ${active(link.href) ? "mobile-nav-link-active" : ""}`}
              >
                <span>{link.label}</span>
                <span aria-hidden="true">›</span>
              </Link>
            ))}
            {session && (
              <button type="button" onClick={signOut} className="mobile-nav-link">
                <span>Sign out</span>
                <span aria-hidden="true">›</span>
              </button>
            )}
          </nav>
          <p className="mobile-nav-foot">
            Private, local academic intelligence
            <br />
            for focused learning and assessment.
          </p>
        </aside>
      </div>
    </>
  );
}
