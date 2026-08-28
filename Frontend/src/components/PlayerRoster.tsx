import { useEffect, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import Chip from "@mui/material/Chip";
import Stack from "@mui/material/Stack";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";
import { api } from "../lib/api";
import { LoadingSpinner } from "./LoadingSpinner";

// A roster of real people's names, kept independent of any one video so
// it's a dropdown pick instead of a retype every time a new job needs
// naming - see PlayerGroupingEditor's Autocomplete, which reads this same
// list and adds to it whenever a genuinely new name is typed there.
export function PlayerRoster() {
  const [roster, setRoster] = useState<string[] | null>(null);
  const [input, setInput] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .getRoster()
      .then((res) => !cancelled && setRoster(res.players))
      .catch((err) => !cancelled && setError(err instanceof Error ? err.message : String(err)));
    return () => {
      cancelled = true;
    };
  }, []);

  async function handleAdd() {
    const name = input.trim();
    if (!name) return;
    setError(null);
    try {
      const res = await api.addToRoster(name);
      setRoster(res.players);
      setInput("");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleRemove(name: string) {
    setError(null);
    try {
      const res = await api.removeFromRoster(name);
      setRoster(res.players);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <Card variant="outlined" sx={{ p: 2.5, mb: 5 }}>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 0.5 }}>
        Players
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Names added here (or while naming players on any video) are remembered across every video, so
        you can pick from a dropdown instead of retyping them each time.
      </Typography>

      <Stack direction="row" spacing={1} sx={{ mb: 2, maxWidth: 420 }}>
        <TextField
          fullWidth
          size="small"
          placeholder="Add a player's name"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") void handleAdd();
          }}
        />
        <Button variant="contained" onClick={() => void handleAdd()}>
          Add
        </Button>
      </Stack>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {roster === null ? (
        <LoadingSpinner minHeight={60} />
      ) : (
        <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
          {roster.length === 0 && (
            <Typography variant="body2" color="text.secondary">
              No players added yet.
            </Typography>
          )}
          {roster.map((name) => (
            <Chip key={name} label={name} onDelete={() => void handleRemove(name)} />
          ))}
        </Box>
      )}
    </Card>
  );
}
