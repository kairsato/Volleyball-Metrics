import { useEffect, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import FormControlLabel from "@mui/material/FormControlLabel";
import IconButton from "@mui/material/IconButton";
import InputAdornment from "@mui/material/InputAdornment";
import Link from "@mui/material/Link";
import MenuItem from "@mui/material/MenuItem";
import Paper from "@mui/material/Paper";
import Stack from "@mui/material/Stack";
import Switch from "@mui/material/Switch";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import ShuffleIcon from "@mui/icons-material/Shuffle";
import VisibilityIcon from "@mui/icons-material/Visibility";
import VisibilityOffIcon from "@mui/icons-material/VisibilityOff";
import { api } from "../lib/api";
import type { AuthStatus, ShareStatus } from "../lib/types";

// Everything needed to make this app reachable outside the local network:
// a login password (typed or generated, off by default - see
// Backend/API/auth.py's module docstring), an optional UPnP toggle to have
// the router open the needed ports automatically, and a copyable share URL.
// The login gate itself lives in LoginGate.tsx, which picks up this
// panel's changes on its very next status check. Lives on SettingsPage's
// Share tab - previously a dialog off the gear menu, moved here so Share/
// Configuration/Debug all live in one place.
function formatExpiryTimestamp(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function formatRemaining(iso: string): string {
  const totalMinutes = Math.max(0, Math.floor((new Date(iso).getTime() - Date.now()) / 60_000));
  const days = Math.floor(totalMinutes / (60 * 24));
  const hours = Math.floor((totalMinutes % (60 * 24)) / 60);
  if (days > 0) return `${days} day${days === 1 ? "" : "s"} ${hours} hr${hours === 1 ? "" : "s"}`;
  const minutes = totalMinutes % 60;
  return `${hours} hr${hours === 1 ? "" : "s"} ${minutes} min${minutes === 1 ? "" : "s"}`;
}

export function ShareSettingsPanel() {
  const [status, setStatus] = useState<AuthStatus | null>(null);
  const [password, setPassword] = useState("");
  const [passwordVisible, setPasswordVisible] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const [share, setShare] = useState<ShareStatus | null>(null);
  const [upnpSaving, setUpnpSaving] = useState(false);
  const [urlRevealed, setUrlRevealed] = useState(false);
  const [hostnameInput, setHostnameInput] = useState("");
  const [hostnameSaving, setHostnameSaving] = useState(false);
  // Turning ports off is a real router change, not just a UI flip - confirm
  // before actually doing it rather than closing them on the first click.
  const [confirmUpnpOff, setConfirmUpnpOff] = useState(false);
  // How long Share stays on before auto-disabling itself once switched on -
  // null means "Forever" (never auto-expires). Only read at the moment the
  // switch turns on (see handleToggleEnabled) - not persisted/restored.
  const [durationDays, setDurationDays] = useState<number | null>(7);

  useEffect(() => {
    api.authStatus().then(setStatus).catch(() => undefined);
    api
      .shareStatus()
      .then((s) => {
        setShare(s);
        setHostnameInput(s.hostname ?? "");
      })
      .catch(() => undefined);
  }, []);

  async function handleSetPassword() {
    setSaving(true);
    setError(null);
    setMessage(null);
    try {
      const res = await api.authSetPassword(password);
      setStatus(res);
      setPassword("");
      setPasswordVisible(false);
      setMessage("Password updated.");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  async function handleSuggestPassword() {
    setError(null);
    try {
      const res = await api.authSuggestPassword();
      setPassword(res.password);
      setPasswordVisible(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleToggleEnabled(enabled: boolean) {
    setSaving(true);
    setError(null);
    setMessage(null);
    try {
      setStatus(await api.authSetEnabled(enabled, durationDays));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  async function handleToggleUpnp(enabled: boolean) {
    setUpnpSaving(true);
    setError(null);
    try {
      setShare(enabled ? await api.enableUpnp() : await api.disableUpnp());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setUpnpSaving(false);
    }
  }

  async function handleSetHostname() {
    setHostnameSaving(true);
    setError(null);
    setMessage(null);
    try {
      const res = await api.setHostname(hostnameInput.trim() || null);
      setShare(res);
      setHostnameInput(res.hostname ?? "");
      setMessage(res.hostname ? "Hostname saved." : "Hostname cleared.");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setHostnameSaving(false);
    }
  }

  function copyToClipboard(text: string) {
    navigator.clipboard?.writeText(text).catch(() => undefined);
  }

  return (
    <Box sx={{ maxWidth: 720 }}>
      <Stack direction="row" sx={{ alignItems: "center", justifyContent: "space-between", mb: 2 }}>
        <Typography variant="h6">Share</Typography>
        <FormControlLabel
          labelPlacement="start"
          sx={{ mr: 0 }}
          control={
            <Switch
              checked={status?.enabled ?? false}
              disabled={saving || !status?.password_set}
              onChange={(event) => void handleToggleEnabled(event.target.checked)}
            />
          }
          label="Enable Share"
        />
      </Stack>

      <Stack spacing={3}>
        <Alert severity="warning">
          Enabling Share makes this app reachable from the internet. Your password is the only thing standing
          between a stranger and your videos - use a strong one (ideally generated). Share settings themselves
          (this page, and changing the password) stay locked to your local network either way - only the rest of
          the app is reachable remotely.
        </Alert>

        <TextField
          select
          size="small"
          label="Share stays on for"
          value={durationDays === null ? "forever" : String(durationDays)}
          onChange={(event) => setDurationDays(event.target.value === "forever" ? null : Number(event.target.value))}
          disabled={saving}
          sx={{ maxWidth: 220 }}
        >
          <MenuItem value="1">1 day</MenuItem>
          <MenuItem value="7">7 days</MenuItem>
          <MenuItem value="forever">Forever</MenuItem>
        </TextField>

        <Paper variant="outlined" sx={{ p: 2, borderRadius: 2 }}>
          <Stack spacing={2}>
            <Typography variant="subtitle2">Password</Typography>
            <Typography variant="body2" color="text.secondary">
              Off by default. Allows outside users to access this site, but requires authentication (a password) to
              do so.
            </Typography>

            <TextField
              type={passwordVisible ? "text" : "password"}
              label={status?.password_set ? "New password" : "Set a password"}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              helperText="At least 16 characters, with at least 3 of: lowercase, uppercase, digits, symbols."
              fullWidth
              slotProps={{
                input: {
                  endAdornment: (
                    <InputAdornment position="end">
                      <IconButton
                        size="small"
                        aria-label="Generate a strong password"
                        onClick={() => void handleSuggestPassword()}
                      >
                        <ShuffleIcon fontSize="small" />
                      </IconButton>
                      <IconButton
                        size="small"
                        aria-label={passwordVisible ? "Hide password" : "Show password"}
                        onClick={() => setPasswordVisible((v) => !v)}
                      >
                        {passwordVisible ? <VisibilityOffIcon fontSize="small" /> : <VisibilityIcon fontSize="small" />}
                      </IconButton>
                      <IconButton
                        size="small"
                        aria-label="Copy password"
                        disabled={!password}
                        onClick={() => copyToClipboard(password)}
                      >
                        <ContentCopyIcon fontSize="small" />
                      </IconButton>
                    </InputAdornment>
                  ),
                },
              }}
            />
            <Button variant="outlined" disabled={saving || !password} onClick={() => void handleSetPassword()}>
              {status?.password_set ? "Update password" : "Set password"}
            </Button>

            {!status?.password_set && (
              <Typography variant="caption" color="text.secondary">
                Set a password above before turning this on.
              </Typography>
            )}
            {status?.enabled && (
              <Typography variant="caption" color="text.secondary">
                {status.expires_at
                  ? `Expires ${formatExpiryTimestamp(status.expires_at)} (${formatRemaining(status.expires_at)} remaining).`
                  : "Set to never expire automatically - turn it off yourself when you're done."}
              </Typography>
            )}
          </Stack>
        </Paper>

        <Paper variant="outlined" sx={{ p: 2, borderRadius: 2 }}>
          <Stack spacing={1.5}>
            <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
              <Typography variant="subtitle2">Port forwarding</Typography>
              <Chip label="Required" size="small" color="warning" variant="outlined" />
            </Stack>

            <Alert severity="info" sx={{ py: 0.5 }}>
              Some ISPs use CGNAT by default, which silently stops port forwarding from working at all - visit your
              ISP's website to check if this applies to you.
            </Alert>

            {share && (
              <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap" }}>
                {share.ports.map((p) => (
                  <Chip
                    key={p.port}
                    label={`Port ${p.port} · TCP${p.status === "open" ? " · Open" : p.status === "error" ? " · Error" : ""}`}
                    size="small"
                    color={p.status === "open" ? "success" : "error"}
                    variant={p.status === "open" ? "filled" : "outlined"}
                  />
                ))}
              </Stack>
            )}

            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              Methods
            </Typography>

            <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
              <Stack spacing={1} sx={{ flex: 1 }}>
                <Typography variant="body2" sx={{ fontWeight: 600 }}>
                  1. Automatic (UPnP)
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Your router must support UPnP and have it enabled.
                </Typography>
                <FormControlLabel
                  control={
                    <Switch
                      checked={share?.upnp_enabled ?? false}
                      disabled={upnpSaving || !status?.enabled}
                      onChange={(event) =>
                        event.target.checked ? void handleToggleUpnp(true) : setConfirmUpnpOff(true)
                      }
                    />
                  }
                  label="Automatically open ports"
                />
                {!status?.enabled && (
                  <Typography variant="caption" color="text.secondary">
                    Turn on Enable Share above first.
                  </Typography>
                )}
                {share?.last_error && (
                  <Typography variant="caption" color="error">
                    {share.last_error}
                  </Typography>
                )}
              </Stack>

              <Stack spacing={1} sx={{ flex: 1 }}>
                <Typography variant="body2" sx={{ fontWeight: 600 }}>
                  2. Manual
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Forward the ports above yourself in your router's settings.{" "}
                  <Link href="https://portforward.com" target="_blank" rel="noopener noreferrer">
                    portforward.com
                  </Link>{" "}
                  has router-specific guides if you're not sure how.
                </Typography>
              </Stack>
            </Stack>
          </Stack>
        </Paper>

        <Paper variant="outlined" sx={{ p: 2, borderRadius: 2 }}>
          <Stack spacing={1}>
            <Typography variant="subtitle2">Custom domain (optional)</Typography>
            <Typography variant="body2" color="text.secondary">
              Point a domain at this machine - a free dynamic-DNS one (e.g. via DuckDNS) works fine if your IP isn't
              static - and set it here. The Share URL below uses it instead of a raw IP, and if you're running this
              behind the bundled Caddy reverse proxy (see ../Caddyfile), Caddy picks it up automatically next time
              it starts, so it can get a real, browser-trusted HTTPS certificate for it.
            </Typography>
            <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
              <TextField
                size="small"
                fullWidth
                placeholder="example.duckdns.org"
                value={hostnameInput}
                onChange={(event) => setHostnameInput(event.target.value)}
                disabled={hostnameSaving}
              />
              <Button
                variant="outlined"
                disabled={hostnameSaving || hostnameInput.trim() === (share?.hostname ?? "")}
                onClick={() => void handleSetHostname()}
              >
                Save
              </Button>
            </Stack>
          </Stack>
        </Paper>

        <Paper variant="outlined" sx={{ p: 2, borderRadius: 2 }}>
          <Stack spacing={1}>
            <Typography variant="subtitle2">Share URL</Typography>
            <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
              <TextField
                size="small"
                fullWidth
                value={share ? (urlRevealed ? share.share_url : "•".repeat(share.share_url.length)) : ""}
                slotProps={{ input: { readOnly: true } }}
              />
              <IconButton
                aria-label={urlRevealed ? "Hide share URL" : "Show share URL"}
                onClick={() => setUrlRevealed((v) => !v)}
              >
                {urlRevealed ? <VisibilityOffIcon fontSize="small" /> : <VisibilityIcon fontSize="small" />}
              </IconButton>
              <IconButton aria-label="Copy share URL" onClick={() => share && copyToClipboard(share.share_url)}>
                <ContentCopyIcon fontSize="small" />
              </IconButton>
            </Stack>
            <Typography variant="caption" color="text.secondary">
              Hidden by default since it includes this device's network address.
            </Typography>
          </Stack>
        </Paper>

        {message && <Alert severity="success">{message}</Alert>}
        {error && <Alert severity="error">{error}</Alert>}
      </Stack>

      <Dialog open={confirmUpnpOff} onClose={() => setConfirmUpnpOff(false)} maxWidth="xs" fullWidth>
        <DialogTitle>Turn off automatic port forwarding?</DialogTitle>
        <DialogContent>
          <Typography variant="body2">
            This will close {share?.ports.map((p) => p.port).join(", ") ?? "these"} on your router. You can turn it
            back on anytime.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmUpnpOff(false)}>Cancel</Button>
          <Button
            color="warning"
            onClick={() => {
              setConfirmUpnpOff(false);
              void handleToggleUpnp(false);
            }}
          >
            Turn off
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
