"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

const links = [
  { href: "/", label: "Home" },
  { href: "/student", label: "Student" },
  { href: "/teacher", label: "Teacher" },
];

export default function AppHeader() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  
  const active = (href: string) => href === "/" ? pathname === href : pathname.startsWith(href);

  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => event.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, []);

  return <>
    <header className="sticky top-0 z-40 mx-auto w-[min(1260px,100%)] px-5 pt-6 sm:px-8 sm:pt-8">
      <div className="flex items-center justify-between gap-4">
        {/* Floating Brand */}
    <Link
  href="/"
  onClick={() => setOpen(false)}
  className="floating-brand ml-2 flex shrink-0 items-center gap-2 text-lg font-bold tracking-[-.05em] text-[#16213d] transition-transform hover:scale-105"
>
  <span className="grid h-7 w-7 place-items-center">
    <img
      src="/framework-icon.svg"
      alt="Framework"
      className="h-7 w-7"
    />
  </span>
  EduGuard
</Link>

        {/* Floating Navigation Dock */}
        <nav aria-label="Primary navigation" className="floating-nav-dock">
          <div className="floating-nav-inner">
            {links.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                onClick={() => setOpen(false)}
                className={`floating-nav-link ${active(link.href) ? 'floating-nav-link-active' : ''}`}
                aria-current={active(link.href) ? 'page' : undefined}
              >
                {link.label}
              </Link>
            ))}
          </div>
        </nav>

        {/* Mobile Menu Button */}
        <button 
          className="mobile-menu-button" 
          aria-expanded={open} 
          aria-controls="mobile-navigation" 
          aria-label="Open navigation menu" 
          onClick={() => setOpen(true)}
        >
          <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round">
            <path d="M4 7h16M4 12h16M4 17h16" />
          </svg>
          <span className="sr-only">Menu</span>
        </button>
      </div>
    </header>
    
    {/* Mobile Navigation */}
    <div id="mobile-navigation" className={`mobile-nav ${open ? "mobile-nav-open" : ""}`} aria-hidden={!open}>
      <button className="mobile-nav-backdrop" onClick={() => setOpen(false)} aria-label="Close navigation menu" tabIndex={open ? 0 : -1} />
      <aside className="mobile-nav-panel" role="dialog" aria-modal="true" aria-label="Framework navigation">
        <div className="mobile-nav-heading">
          <div>
            <span className="mobile-nav-kicker">EduGuard</span>
            <h2>Explore your workspace</h2>
          </div>
          <button onClick={() => setOpen(false)} className="mobile-nav-close" aria-label="Close navigation menu">
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
              className={`mobile-nav-link ${active(link.href) ? "mobile-nav-link-active" : ""}`}
            >
              <span>{link.label}</span>
              <span aria-hidden="true">›</span>
            </Link>
          ))}
        </nav>
        <p className="mobile-nav-foot">Private, local academic intelligence<br />for focused learning and assessment.</p>
      </aside>
    </div>
  </>;
}
