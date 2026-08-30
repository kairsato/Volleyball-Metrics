import { useState } from "react";
import Button from "@mui/material/Button";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogContentText from "@mui/material/DialogContentText";
import DialogTitle from "@mui/material/DialogTitle";
import RestartAltIcon from "@mui/icons-material/RestartAlt";

interface RedoButtonProps {
  label: string;
  confirmTitle: string;
  confirmText: string;
  onConfirm: () => Promise<void>;
  fullWidth?: boolean;
}

// A second, explicit confirmation is always required before either redo
// action runs - this reprocesses the video, which isn't something a stray
// click should trigger. Shared by the video Setup page's Court and
// Unidentified players tabs (each reprocesses the video differently, but
// the confirm-then-run shape is identical).
export function RedoButton({ label, confirmTitle, confirmText, onConfirm, fullWidth }: RedoButtonProps) {
  const [open, setOpen] = useState(false);
  const [running, setRunning] = useState(false);

  async function handleConfirm() {
    setRunning(true);
    try {
      await onConfirm();
      setOpen(false);
    } finally {
      setRunning(false);
    }
  }

  return (
    <>
      <Button
        variant="outlined"
        color="warning"
        startIcon={<RestartAltIcon />}
        fullWidth={fullWidth}
        onClick={() => setOpen(true)}
      >
        {label}
      </Button>

      <Dialog open={open} onClose={() => (running ? undefined : setOpen(false))}>
        <DialogTitle>{confirmTitle}</DialogTitle>
        <DialogContent>
          <DialogContentText>{confirmText}</DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpen(false)} disabled={running}>
            Cancel
          </Button>
          <Button color="warning" variant="contained" onClick={() => void handleConfirm()} disabled={running}>
            {running ? "Working..." : "Confirm"}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}
