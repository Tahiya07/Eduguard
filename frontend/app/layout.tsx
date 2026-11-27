import "./globals.css";
import AppHeader from "./app-header";
import PwaInstall from "./pwa-install";

export const metadata = {
  title: "EduGuard · Academic assistance",
  description: "Private, local Bloom-aware academic assistance",
  manifest: "/manifest.webmanifest",
  icons: { icon: "/framework-icon.svg" },
};
export const viewport = { width: "device-width", initialScale: 1, viewportFit: "cover", themeColor: "#294f7b" };

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="app-shell">
          <AppHeader />
          {children}
          <PwaInstall />
        </div>
      </body>
    </html>
  );
}
