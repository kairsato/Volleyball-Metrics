import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Binds to all network interfaces (0.0.0.0), not just localhost, so
    // another device on the same LAN can load the dev server via this
    // machine's own IP (e.g. http://192.168.1.27:5173) - not just from
    // this machine itself.
    host: true,
  },
})
