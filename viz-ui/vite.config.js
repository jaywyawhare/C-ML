import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const API_TARGET = 'http://localhost:8001';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      // Handle web-worker import from elkjs
      'web-worker': 'web-worker',
    },
  },
  optimizeDeps: {
    include: ['web-worker'],
  },
  server: {
    port: 5173,
    host: true,
    proxy: {
      // Data endpoints
      '/graph': { target: API_TARGET, changeOrigin: true },
      '/training': { target: API_TARGET, changeOrigin: true },
      '/model_architecture': { target: API_TARGET, changeOrigin: true },
      '/kernels': { target: API_TARGET, changeOrigin: true },
      // Utility endpoints
      '/ctxs': { target: API_TARGET, changeOrigin: true },
      '/health': { target: API_TARGET, changeOrigin: true },
      '/status': { target: API_TARGET, changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true,
        pure_funcs: ['console.log', 'console.info', 'console.debug'],
      },
    },
    chunkSizeWarningLimit: 1000,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          charts: ['recharts'],
          graph: ['cytoscape', 'cytoscape-dagre', 'cytoscape-elk', 'elkjs'],
          syntax: ['react-syntax-highlighter'],
        },
      },
    },
  },
});
