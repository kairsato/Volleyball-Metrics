import type { ActionQualityCategory } from "../../lib/types";

export type ActionCategoryKey = "serve" | "receive" | "set" | "spike";

// Only factors a player can actually do something about get a tip - "time
// given" (set) reflects how good the pass before it was, and "blockers"
// (spike) reflects the opponent's positioning, not the hitter's own
// execution, so both are deliberately left out here even though they're
// real scored factors elsewhere.
const TIPS: Record<ActionCategoryKey, Record<string, string>> = {
  serve: {
    speed: "Work on generating more serve speed - a stronger arm swing or jump serve can add pace.",
    placement: "Aim further from the nearest opponent - target seams and open zones instead of straight at them.",
    trajectory_height: "Keep serves lower over the net - a flatter trajectory gives the defense less time to react.",
  },
  receive: {
    placement: "Work on directing the pass more precisely toward the target zone.",
    reaction: "Get lower and set your platform earlier - reaction time is lagging the ball's approach.",
    handling_speed: "Absorb more pace on contact - the ball is coming off the platform faster than ideal.",
  },
  set: {
    placement: "Work on consistently delivering sets to the target zone for the hitters.",
    height: "Aim for a more consistent set height - not too flat, not too high.",
  },
  spike: {
    positioning: "Approach closer to the net before attacking to cut down the angle.",
    speed: "Work on approach/arm speed to hit with more pace.",
    placement: "Aim away from where defenders are positioned rather than straight at them.",
  },
};

function average(values: number[]): number | null {
  return values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : null;
}

// Below this gap (the category-wide average minus this player's own
// average, both on the factor's normal 0-1 scale), a factor isn't called
// out - keeps evenly-matched players from getting a noisy "improve this"
// tag over a difference that isn't really meaningful.
const MIN_GAP = 0.08;

// The single most notable thing a player could work on in this category,
// or null if nothing clears MIN_GAP (or there's no data for them at all) -
// found by comparing their own average on each actionable factor against
// the category-wide average on that same factor, and surfacing the
// largest shortfall.
export function playerInsight(category: ActionQualityCategory, categoryKey: ActionCategoryKey, stableId: number): string | null {
  const tips = TIPS[categoryKey];
  const playerInstances = category.instances.filter((i) => i.player_stable_id === stableId);
  if (playerInstances.length === 0) return null;

  let worstFactor: string | null = null;
  let worstGap = MIN_GAP;

  for (const factorKey of Object.keys(tips)) {
    const playerValues = playerInstances
      .map((i) => i.factors[factorKey])
      .filter((v): v is number => v !== null && v !== undefined);
    if (playerValues.length === 0) continue;

    const allValues = category.instances
      .map((i) => i.factors[factorKey])
      .filter((v): v is number => v !== null && v !== undefined);

    const overallAvg = average(allValues);
    const playerAvg = average(playerValues);
    if (overallAvg === null || playerAvg === null) continue;

    const gap = overallAvg - playerAvg;
    if (gap > worstGap) {
      worstGap = gap;
      worstFactor = factorKey;
    }
  }

  return worstFactor ? tips[worstFactor] : null;
}
