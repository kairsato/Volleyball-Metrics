import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import Stack from "@mui/material/Stack";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";
import { api } from "../lib/api";
import type { Captcha } from "../lib/types";

interface LoginGateProps {
  children: ReactNode;
}

// Gates the whole app behind a login screen only when the backend actually
// has login turned on (see Backend/API/auth.py - off by default, matching
// how this app has always run with no auth at all) AND says THIS browser
// specifically needs it (status.login_required, not status.enabled) - a
// LAN visitor never needs to log in, even while Share has login on for
// remote ones (see main.py's AuthMiddleware, which has the same LAN
// bypass this mirrors). Also listens for api.ts's "auth:unauthorized"
// event, dispatched whenever any API call gets a 401, so an expired/
// revoked session drops back to this screen mid-use instead of the app
// just silently failing every request after that.
//
// Renders the requested page optimistically (assuming no login is needed)
// rather than blocking on the initial status check first - the common case
// (login off, or already logged in) never sees a login screen flash in
// front of the page it asked for. Only swaps to the login screen once the
// check (or a live 401) actually confirms it's required.
export function LoginGate({ children }: LoginGateProps) {
  const [needsLogin, setNeedsLogin] = useState(false);
  const [captcha, setCaptcha] = useState<Captcha | null>(null);
  const [password, setPassword] = useState("");
  const [captchaAnswer, setCaptchaAnswer] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  function checkStatus() {
    return api
      .authStatus()
      .then((status) => setNeedsLogin(status.login_required))
      .catch(() => setNeedsLogin(false));
  }

  useEffect(() => {
    void checkStatus();
    // checkStatus is stable in the sense that matters here (it doesn't
    // close over any state that changes its behavior) - omitted from deps
    // to avoid re-running this on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    function handleUnauthorized() {
      // A real 401 can only happen while login is actually enabled server-
      // side (see AuthMiddleware), so this can go straight to the login
      // screen without re-checking status first.
      setNeedsLogin(true);
    }
    // Logging out, unlike a 401, can happen whether or not login is even
    // enabled - re-check status rather than assuming a login screen is
    // warranted, so "log out" is a no-op when login was never turned on.
    function handleLogout() {
      void checkStatus();
    }
    window.addEventListener("auth:unauthorized", handleUnauthorized);
    window.addEventListener("auth:logout", handleLogout);
    return () => {
      window.removeEventListener("auth:unauthorized", handleUnauthorized);
      window.removeEventListener("auth:logout", handleLogout);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function loadCaptcha() {
    setCaptcha(null);
    setCaptchaAnswer("");
    api.authGetCaptcha().then(setCaptcha).catch(() => undefined);
  }

  useEffect(() => {
    if (needsLogin) loadCaptcha();
  }, [needsLogin]);

  async function handleSubmit() {
    if (!captcha) return;
    setSubmitting(true);
    setError(null);
    try {
      await api.authLogin(password, captcha.captcha_id, captchaAnswer);
      setNeedsLogin(false);
      setPassword("");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      loadCaptcha();
    } finally {
      setSubmitting(false);
    }
  }

  if (!needsLogin) return <>{children}</>;

  return (
    <Box sx={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", p: 2 }}>
      <Card variant="outlined" sx={{ p: 4, width: "100%", maxWidth: 380 }}>
        <Typography variant="h5" sx={{ fontWeight: 700, mb: 3 }}>
          🏐 Volleyball Metrics
        </Typography>

        <Stack
          component="form"
          spacing={2}
          onSubmit={(event) => {
            event.preventDefault();
            void handleSubmit();
          }}
        >
          <TextField
            type="password"
            label="Password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            autoFocus
            fullWidth
          />

          {captcha && (
            <Box
              component="img"
              src={`data:image/png;base64,${captcha.image_base64}`}
              alt="Captcha challenge"
              sx={{ borderRadius: 1, border: 1, borderColor: "divider", alignSelf: "center" }}
            />
          )}

          <Stack direction="row" spacing={1}>
            <TextField
              label="Enter the code above"
              value={captchaAnswer}
              onChange={(event) => setCaptchaAnswer(event.target.value)}
              fullWidth
            />
            <Button onClick={loadCaptcha} disabled={submitting} sx={{ flexShrink: 0 }}>
              New code
            </Button>
          </Stack>

          {error && <Alert severity="error">{error}</Alert>}

          <Button
            type="submit"
            variant="contained"
            disabled={submitting || !password || !captchaAnswer || !captcha}
          >
            {submitting ? "Signing in..." : "Sign in"}
          </Button>
        </Stack>
      </Card>
    </Box>
  );
}
