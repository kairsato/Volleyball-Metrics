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
