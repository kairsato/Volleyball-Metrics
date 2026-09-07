// Shared by every time-synced video overlay (BallMinimap, BallTrackingOverlay,
// PlayerTrackingOverlay) - each one's data arrives decimated (see the
// backend's BALL_TRAJECTORY_STRIDE/PLAYER_TRAJECTORY_STRIDE) rather than at
// full per-frame resolution, so "what to show right now" only actually
// changes a few times a second even though the video's own timeupdate fires
// far more often than that. BallMinimap/BallTrackingOverlay use this to find
// the current index in the PARENT (VideoPlayer), not inside themselves,
// specifically so that index - a plain number - can be passed down as a
// prop to a React.memo'd component: unless the index itself changes, memo
// correctly skips re-rendering the overlay's SVG content on every single
// timeupdate tick. PlayerTrackingOverlay uses it differently (see its own
// doc comment) since it interpolates between samples rather than snapping.

// Last item at or before `t`, by binary search - `items` must already be
// sorted by `.t` ascending (the order every /*-trajectory endpoint returns
// its points/frames in).
export function findIndexAtOrBefore<T extends { t: number }>(items: T[], t: number): number {
  let lo = 0;
  let hi = items.length - 1;
  let result = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (items[mid].t <= t) {
      result = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return result;
}
