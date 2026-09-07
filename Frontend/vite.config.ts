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
    // Vite rejects requests whose Host header it doesn't recognize (DNS
    // rebinding protection) - fine for a LAN IP, but the Caddy reverse
    // proxy in front of this (see ../Caddyfile) forwards the ORIGINAL Host
    // header through unchanged, so a request arriving via a public domain
    // (whatever hostname was set in the app's Share settings) still says
    // e.g. "Host: example.duckdns.org" by the time it reaches here. `true`
    // disables the check entirely rather than listing a specific hostname -
    // safe here since this dev server is never exposed directly, only
    // through Caddy (see ../Caddyfile), which already only answers for the
    // one hostname it's configured with.
    allowedHosts: true,
  },
})
