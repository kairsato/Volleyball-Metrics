import type { PlayerEvent } from "../../lib/types";

export interface FlatEvent extends PlayerEvent {
  playerId: string;
  playerName: string;
}

export function formatTimestamp(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}
