import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "QuantumMeta-Sat — Interactive Signal Simulator",
  description: "See exactly how 'Hello World' travels from ground station through a LEO satellite channel and gets recovered using Quantum Meta-Learning",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet" />
      </head>
      <body>{children}</body>
    </html>
  );
}
