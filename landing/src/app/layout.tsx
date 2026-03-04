import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AEON — The Room That Thinks",
  description:
    "An AI-powered dorm room assistant that sees, hears, and responds.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="antialiased">{children}</body>
    </html>
  );
}
