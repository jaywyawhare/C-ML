import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    proxy: {
      '/graph': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
      '/ctxs': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
      '/training': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
    },
  },
});
