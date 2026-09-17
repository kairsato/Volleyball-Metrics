import type { ActionQualityCategory } from "../../lib/types";

// Mirrors Backend/API/action_quality.py's _weighted_score exactly - a
// missing factor drops out and the remaining weights renormalize, rather
// than penalizing or failing outright. Kept in lockstep with that function
// deliberately so a user's adjusted weights recompute the *same* way the
// server's own defaults do, just with different numbers.
export function weightedScore(factors: Record<string, number | null | undefined>, weights: Record<string, number>): number | null {
  let total = 0;
  let totalWeight = 0;
  for (const [name, weight] of Object.entries(weights)) {
    const value = factors[name];
    if (value === null || value === undefined) continue;
    total += value * weight;
    totalWeight += weight;
  }
  return totalWeight > 0 ? total / totalWeight : null;
}

function average(values: number[]): number | null {
  return values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : null;
}

export function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// Keeps a category's weights summing to exactly 1 as one slider moves -
// the changed factor takes the dragged value outright, and every other
// factor is rescaled to fill the remaining share in the same proportions
// they already had relative to each other (split evenly among them if
// they'd all been at 0). This is what lets AnalyticsTab's sliders behave
// as one compositional (100%-total) group instead of independent 0-1 bars.
export function redistributeWeights(
  weights: Record<string, number>,
  changedKey: string,
  rawNewValue: number,
): Record<string, number> {
  const newValue = Math.max(0, Math.min(1, rawNewValue));
  const otherKeys = Object.keys(weights).filter((k) => k !== changedKey);
  const remaining = 1 - newValue;
  const othersSum = otherKeys.reduce((sum, k) => sum + weights[k], 0);

  const result: Record<string, number> = { [changedKey]: newValue };
  for (const key of otherKeys) {
    result[key] = othersSum > 0 ? (weights[key] / othersSum) * remaining : remaining / otherKeys.length;
  }
  return result;
}

export interface RecomputedPlayer {
  averageScore: number | null;
  count: number;
}

export interface RecomputedCategory {
  averageScore: number | null;
  // stable_id -> recomputed average, mirrors ActionQualityCategory.players
  // but reflects the current (possibly user-adjusted) weights.
  players: Record<number, RecomputedPlayer>;
  // One score per instance, same order/index as category.instances.
  instanceScores: (number | null)[];
}

// Client-side re-run of the server's own weighting, driven by whatever
// weights the user currently has dialed in (see the sliders in
// AnalyticsTab) - entirely local, no request to the server at all.
export function recomputeCategory(category: ActionQualityCategory, weights: Record<string, number>): RecomputedCategory {
  const scoresByPlayer = new Map<number, number[]>();
  const allScores: number[] = [];

  const instanceScores = category.instances.map((instance) => {
    const score = weightedScore(instance.factors, weights);
    if (score !== null) {
      allScores.push(score);
      if (instance.player_stable_id !== null) {
        const existing = scoresByPlayer.get(instance.player_stable_id) ?? [];
        existing.push(score);
        scoresByPlayer.set(instance.player_stable_id, existing);
      }
    }
    return score;
  });

  const players: Record<number, RecomputedPlayer> = {};
  for (const [stableId, scores] of scoresByPlayer) {
    players[stableId] = { averageScore: average(scores), count: scores.length };
  }

  return { averageScore: average(allScores), players, instanceScores };
}
