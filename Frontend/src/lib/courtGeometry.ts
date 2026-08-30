import type { Point, PredictedCourtGeometry } from "./types";

// Mirrors Backend/Analysis/CourtDefinition/court.py's
// create_half_court_homography/predict_court_geometry - see that module's
// diagram for the full layout. Reimplemented client-side (rather than
// round-tripping to the backend on every drag) so the near baseline and
// both attack lines redraw live as the user moves any of the 4 points,
// not just after Save. Verified against the same synthetic ground-truth
// perspective transform the Python version is tested with, to
// sub-millimeter-equivalent precision.
const COURT_WIDTH = 9.0;
const COURT_LENGTH = 18.0;
const ATTACK_LINE_OFFSET_M = 3.0;

type Matrix3x3 = [[number, number, number], [number, number, number], [number, number, number]];

// Solves the 8-unknown DLT system (h22 fixed at 1) for the perspective
// transform mapping src -> dst, given exactly 4 point correspondences -
// the same problem cv2.getPerspectiveTransform solves on the backend, via
// straightforward Gaussian elimination with partial pivoting since there's
// no linear-algebra library already in this frontend worth pulling in for
// one 8x8 solve.
function solveHomography(src: [number, number][], dst: [number, number][]): Matrix3x3 {
  const A: number[][] = [];
  const b: number[] = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i];
    const [X, Y] = dst[i];
    A.push([x, y, 1, 0, 0, 0, -x * X, -y * X]);
    b.push(X);
    A.push([0, 0, 0, x, y, 1, -x * Y, -y * Y]);
    b.push(Y);
  }

  const n = 8;
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(A[row][col]) > Math.abs(A[pivot][col])) pivot = row;
    }
    [A[col], A[pivot]] = [A[pivot], A[col]];
    [b[col], b[pivot]] = [b[pivot], b[col]];

    const pivotVal = A[col][col];
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = A[row][col] / pivotVal;
      for (let k = col; k < n; k++) A[row][k] -= factor * A[col][k];
      b[row] -= factor * b[col];
    }
  }

  const h = b.map((v, i) => v / A[i][i]);
  return [
    [h[0], h[1], h[2]],
    [h[3], h[4], h[5]],
    [h[6], h[7], 1],
  ];
}

function invert3x3(m: Matrix3x3): Matrix3x3 {
  const [[a, b, c], [d, e, f], [g, h, i]] = m;
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  return [
    [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
    [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
    [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
  ];
}

function applyHomography(m: Matrix3x3, x: number, y: number): Point {
  const w = m[2][0] * x + m[2][1] * y + m[2][2];
  return {
    x: (m[0][0] * x + m[0][1] * y + m[0][2]) / w,
    y: (m[1][0] * x + m[1][1] * y + m[1][2]) / w,
  };
}

// Given the 4 points the user actually places (in the fixed order the
// backend expects - see calibration.POINT_NAMES), predicts where the near
// baseline's two corners and both attack lines land in pixel space. Throws
// if the 4 points are degenerate (e.g. collinear) and no homography can be
// solved - callers should catch this and just skip drawing the preview
// rather than crash mid-drag.
export function predictCourtGeometry(
  middleLeft: Point,
  middleRight: Point,
  farLeft: Point,
  farRight: Point,
): PredictedCourtGeometry {
  const dst: [number, number][] = [
    [COURT_LENGTH / 2, 0],
    [COURT_LENGTH / 2, COURT_WIDTH],
    [COURT_LENGTH, 0],
    [COURT_LENGTH, COURT_WIDTH],
  ];
  const src: [number, number][] = [
    [middleLeft.x, middleLeft.y],
    [middleRight.x, middleRight.y],
    [farLeft.x, farLeft.y],
    [farRight.x, farRight.y],
  ];

  const matrix = solveHomography(src, dst);
  const inverse = invert3x3(matrix);

  const nearX = 0;
  const farAttackX = COURT_LENGTH / 2 + ATTACK_LINE_OFFSET_M;
  const nearAttackX = COURT_LENGTH / 2 - ATTACK_LINE_OFFSET_M;

  const toPixel = (cx: number, cy: number) => applyHomography(inverse, cx, cy);

  return {
    near_left: toPixel(nearX, 0),
    near_right: toPixel(nearX, COURT_WIDTH),
    attack_far_left: toPixel(farAttackX, 0),
    attack_far_right: toPixel(farAttackX, COURT_WIDTH),
    attack_near_left: toPixel(nearAttackX, 0),
    attack_near_right: toPixel(nearAttackX, COURT_WIDTH),
  };
}
