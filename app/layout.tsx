import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Analytics } from "@vercel/analytics/next";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const title = "docs-copilot — RAG assistant over the Vercel AI SDK documentation";
const description =
  "Retrieve-then-rerank over the Vercel AI SDK documentation, answering with source attribution and refusing to answer when the docs don't cover the question.";

export const metadata: Metadata = {
  metadataBase: new URL("https://docs-copilot-w89t.vercel.app"),
  title,
  description,
  openGraph: {
    type: "website",
    url: "/",
    siteName: "docs-copilot",
    title,
    description,
    images: [
      {
        url: "/og.png",
        width: 1200,
        height: 630,
        alt: "docs-copilot — a RAG assistant over the Vercel AI SDK documentation",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title,
    description,
    images: ["/og.png"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">
        {children}
        {/* Page-level analytics (visits, referrers, countries). Cookieless; beacons to
            Vercel, not to our API — so it adds no public write surface of our own. */}
        <Analytics />
      </body>
    </html>
  );
}
