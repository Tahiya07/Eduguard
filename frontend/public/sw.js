const CACHE = "framework-shell-v3";
const SHELL = ["/", "/student", "/teacher", "/teacher/exams", "/teacher/corpus", "/settings", "/manifest.webmanifest", "/framework-icon.svg"];
self.addEventListener("install", event => event.waitUntil(caches.open(CACHE).then(cache => cache.addAll(SHELL)).then(() => self.skipWaiting())));
self.addEventListener("activate", event => event.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(key => key !== CACHE).map(key => caches.delete(key)))).then(() => self.clients.claim())));
self.addEventListener("fetch", event => {
  const request = event.request;
  const url = new URL(request.url);
  // Next.js development bundles must always come from the dev server. Caching
  // them can pair new server HTML with an older client component and cause a
  // hydration mismatch after a UI change.
  if (url.origin !== self.location.origin || request.method !== "GET" || url.pathname.startsWith("/api/") || url.pathname.startsWith("/_next/")) return;
  if (request.mode === "navigate") {
    event.respondWith(fetch(request).then(response => { const copy=response.clone(); caches.open(CACHE).then(cache => cache.put(request,copy)); return response; }).catch(() => caches.match(request).then(hit => hit || caches.match("/"))));
    return;
  }
  event.respondWith(caches.match(request).then(hit => hit || fetch(request).then(response => { if (response.ok) { const copy=response.clone(); caches.open(CACHE).then(cache => cache.put(request,copy)); } return response; })));
});
