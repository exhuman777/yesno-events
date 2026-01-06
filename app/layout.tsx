import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "yesno.events - Mention Markets",
  description: "Real-time prediction markets for word frequencies in news cycles",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
