import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Production config - skip TypeScript checking
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    sourcemap: false,
    minify: true,
    rollupOptions: {
      onwarn(warning, warn) {
        // Skip TypeScript warnings during build
        if (warning.code === 'UNUSED_EXTERNAL_IMPORT') return
        if (warning.code === 'CIRCULAR_DEPENDENCY') return
        warn(warning)
      }
    }
  },
  esbuild: {
    // Skip TypeScript checking
    logOverride: { 'this-is-undefined-in-esm': 'silent' }
  }
})
