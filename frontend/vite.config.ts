import path from "path"
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

const BACKEND_URLS: Record<string, string> = {
  development: 'http://127.0.0.1:8000',
  production: 'https://clearpath-rag-873904783482.asia-south1.run.app',
}

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const backendUrl = BACKEND_URLS[mode] || BACKEND_URLS.development

  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
    server: {
      proxy: {
        '/query': backendUrl,
      }
    }
  }
})

