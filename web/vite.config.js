import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base "./" so the same build works on Vercel, Netlify or a static HF Space.
export default defineConfig({
  plugins: [react()],
  base: "./",
});
