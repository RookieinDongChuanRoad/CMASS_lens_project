"use strict";

/**
 * Build a three-slide PowerPoint deck that summarizes the latest production
 * `devauc` and `sersic` results:
 * 1. Posterior corner plots
 * 2. Canonical posterior predictive histograms
 * 3. Fig. 8-like posterior trend summaries
 *
 * Why this script is data-driven instead of hand-placing static strings:
 * - the user is iterating on science outputs quickly
 * - the deck should always point at the current run artifacts on disk
 * - summary text should stay synchronized with JSON contracts such as
 *   `n_posterior_draws_used`, percentile locations, and run IDs
 */

const fs = require("fs");
const path = require("path");
const PptxGenJS = require("pptxgenjs");
const {
  imageSizingContain,
} = require("./pptxgenjs_helpers/image");
const {
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = require("./pptxgenjs_helpers/layout");

const ROOT_DIR = __dirname;
const PROJECT_ROOT = path.resolve(ROOT_DIR, "..", "..");
const OUTPUT_PPTX = path.join(ROOT_DIR, "cmass_profile_summary_20260311.pptx");

/**
 * Centralize all file paths here so future reruns only need one update if the
 * canonical run IDs change. The deck uses the same run pair across all slides.
 */
const PROFILE_RUNS = {
  devauc: {
    key: "devauc",
    label: "de Vaucouleurs",
    shortLabel: "devauc",
    runId: "20260308_215643_devauc_devauc_prod_20260308",
    cornerImage: path.join(
      ROOT_DIR,
      "assets",
      "devauc_posterior_corner_trimmed.png"
    ),
    ppcImage: path.join(
      PROJECT_ROOT,
      "Posterior_predictive_test",
      "results",
      "devauc",
      "20260308_215643_devauc_devauc_prod_20260308",
      "ppc_overview.png"
    ),
    ppcSummary: path.join(
      PROJECT_ROOT,
      "Posterior_predictive_test",
      "results",
      "devauc",
      "20260308_215643_devauc_devauc_prod_20260308",
      "ppc_summary.json"
    ),
    fig8Image: path.join(
      ROOT_DIR,
      "assets",
      "devauc_fig8_like_trimmed.png"
    ),
    fig8Summary: path.join(
      PROJECT_ROOT,
      "Posterior_predictive_test",
      "results",
      "devauc",
      "20260308_215643_devauc_devauc_prod_20260308",
      "fig8_like_summary.json"
    ),
  },
  sersic: {
    key: "sersic",
    label: "Sersic",
    shortLabel: "sersic",
    runId: "20260308_221211_sersic_sersic_prod_20260308",
    cornerImage: path.join(
      ROOT_DIR,
      "assets",
      "sersic_posterior_corner_trimmed.png"
    ),
    ppcImage: path.join(
      PROJECT_ROOT,
      "Posterior_predictive_test",
      "results",
      "sersic",
      "20260308_221211_sersic_sersic_prod_20260308",
      "ppc_overview.png"
    ),
    ppcSummary: path.join(
      PROJECT_ROOT,
      "Posterior_predictive_test",
      "results",
      "sersic",
      "20260308_221211_sersic_sersic_prod_20260308",
      "ppc_summary.json"
    ),
    fig8Image: path.join(
      ROOT_DIR,
      "assets",
      "sersic_fig8_like_trimmed.png"
    ),
    fig8Summary: path.join(
      PROJECT_ROOT,
      "Posterior_predictive_test",
      "results",
      "sersic",
      "20260308_221211_sersic_sersic_prod_20260308",
      "fig8_like_summary.json"
    ),
  },
};

/**
 * Keep the scientific deck visually deliberate but restrained:
 * warm paper background, dark ink text, muted sandstone card borders, and one
 * controlled red-blue accent system that matches the plots without fighting
 * them.
 */
const COLORS = {
  bg: "F5F0E8",
  panel: "FFFDF9",
  panelEdge: "D4C6B2",
  ink: "233446",
  muted: "6E6253",
  warmAccent: "A6513D",
  coolAccent: "587A9B",
  sand: "E6D9C6",
  slate: "D7E0E8",
};

/**
 * Guardrail helpers:
 * - fail early when a required image or JSON file is missing
 * - keep summary formatting consistent across slides
 */
function assertExists(targetPath) {
  if (!fs.existsSync(targetPath)) {
    throw new Error(`Required artifact is missing: ${targetPath}`);
  }
}

function readJson(targetPath) {
  assertExists(targetPath);
  /**
   * Some science-exported summary files serialize `NaN` literally. That is not
   * valid strict JSON, so we normalize it to `null` before parsing.
   */
  const rawText = fs.readFileSync(targetPath, "utf8");
  return JSON.parse(rawText.replace(/\bNaN\b/g, "null"));
}

function pct(value) {
  return `${Number(value).toFixed(1)}%`;
}

function fixed(value, digits = 2) {
  return Number(value).toFixed(digits);
}

function signed(value, digits = 2) {
  const rounded = Number(value).toFixed(digits);
  return value >= 0 ? `+${rounded}` : rounded;
}

/**
 * Extract a small set of narrative-ready PPC diagnostics.
 * These are the most useful slide-callout metrics because they tell the viewer:
 * - whether the observed medians land near the center of the replicated
 *   distribution
 * - which statistic remains visibly tensioned
 * - what canonical production contract generated the result
 */
function buildPpcHighlights(summary) {
  const thetaMedian = summary.statistics.theta_ein.median;
  const sigmaMedian = summary.statistics.sigma.median;
  const sigmaStd = summary.statistics.sigma.std;

  return [
    `theta_E median at ${pct(thetaMedian.left_percentile)} of the replicated distribution.`,
    `sigma median at ${pct(sigmaMedian.left_percentile)}; central location is reproduced well.`,
    `sigma std is the strongest mismatch: observed sits at the ${pct(
      sigmaStd.left_percentile
    )} left-tail percentile.`,
  ];
}

/**
 * The Fig. 8-like JSON stores mass-grid bands for parent / detectable /
 * selected populations. For the slide we only need low-mass and high-mass
 * offsets between `selected` and `parent`, which is compact but still conveys
 * the selection trend.
 */
function summarizeFig8Offsets(summary, quantity) {
  const massCenters = summary.mass_bin_centers;
  const selected = summary.bands[quantity].selected.p50;
  const parent = summary.bands[quantity].parent.p50;

  const validRows = massCenters
    .map((massCenter, index) => ({
      massCenter,
      selected: selected[index],
      parent: parent[index],
    }))
    .filter(
      (row) =>
        Number.isFinite(row.massCenter) &&
        Number.isFinite(row.selected) &&
        Number.isFinite(row.parent)
    );

  if (validRows.length === 0) {
    throw new Error(`No finite Fig. 8-like rows found for quantity: ${quantity}`);
  }

  const low = validRows[0];
  const high = validRows[validRows.length - 1];

  return {
    lowMass: low.massCenter,
    highMass: high.massCenter,
    lowOffset: low.selected - low.parent,
    highOffset: high.selected - high.parent,
  };
}

/**
 * Convert the quantitative offsets into short statements that a slide viewer
 * can parse in seconds. We intentionally keep one line per physical quantity.
 */
function buildFig8Highlights(summary) {
  const m5 = summarizeFig8Offsets(summary, "m5");
  const gamma = summarizeFig8Offsets(summary, "gamma");
  const sigma = summarizeFig8Offsets(summary, "sigma_ap");

  return [
    `m5 selection uplift: ${signed(m5.lowOffset, 2)} -> ${signed(
      m5.highOffset,
      2
    )} dex from low to high mass.`,
    `gamma selection shift: ${signed(gamma.lowOffset, 2)} -> ${signed(
      gamma.highOffset,
      2
    )}; sign tells whether selection pushes the curve above or below parent.`,
    `sigma_ap boost: ${signed(sigma.lowOffset, 1)} -> ${signed(
      sigma.highOffset,
      1
    )} km/s across the mass range.`,
  ];
}

/**
 * Draw the common scientific slide chrome:
 * - full-slide background
 * - compact deck label in the upper-right corner
 * - bold title and smaller subtitle
 */
function addSlideChrome(slide, title, subtitle, tag) {
  slide.background = { color: COLORS.bg };

  slide.addShape("rect", {
    x: 0,
    y: 0,
    w: 13.333,
    h: 0.28,
    line: { color: COLORS.bg, transparency: 100 },
    fill: { color: COLORS.ink },
  });

  slide.addText(title, {
    x: 0.48,
    y: 0.42,
    w: 9.2,
    h: 0.38,
    fontFace: "Avenir Next",
    fontSize: 22,
    bold: true,
    color: COLORS.ink,
    margin: 0,
  });

  slide.addText(subtitle, {
    x: 0.5,
    y: 0.83,
    w: 10.2,
    h: 0.3,
    fontFace: "Avenir Next",
    fontSize: 9,
    color: COLORS.muted,
    margin: 0,
  });

  slide.addShape("rect", {
    x: 10.85,
    y: 0.4,
    w: 2.0,
    h: 0.32,
    line: { color: COLORS.coolAccent, pt: 0.8 },
    fill: { color: COLORS.slate },
  });

  slide.addText(tag, {
    x: 11.0,
    y: 0.46,
    w: 1.7,
    h: 0.16,
    fontFace: "Avenir Next",
    fontSize: 8,
    bold: true,
    color: COLORS.ink,
    margin: 0,
    align: "center",
  });
}

/**
 * Build a reusable side-by-side card. Each slide uses the same base container,
 * but passes different image and text payloads.
 */
function addProfileCard(slide, profile, options) {
  slide.addShape("rect", {
    x: options.x,
    y: options.y,
    w: options.w,
    h: options.h,
    line: { color: COLORS.panelEdge, pt: 1.1 },
    fill: { color: COLORS.panel },
  });

  slide.addShape("rect", {
    x: options.x,
    y: options.y,
    w: options.w,
    h: 0.34,
    line: { color: COLORS.warmAccent, transparency: 100 },
    fill: { color: options.headerFill || COLORS.sand },
  });

  slide.addText(profile.label, {
    x: options.x + 0.18,
    y: options.y + 0.08,
    w: 2.3,
    h: 0.18,
    fontFace: "Avenir Next",
    fontSize: 11,
    bold: true,
    color: COLORS.ink,
    margin: 0,
  });

  slide.addText(profile.runId, {
    x: options.x + options.w - 2.7,
    y: options.y + 0.09,
    w: 2.48,
    h: 0.16,
    fontFace: "Avenir Next",
    fontSize: 7.5,
    color: COLORS.muted,
    margin: 0,
    align: "right",
  });
}

/**
 * Write small multi-line summary text. This helper keeps typography and leading
 * consistent, which matters because the deck mixes dense scientific plots with
 * compact prose.
 */
function addSummaryLines(slide, lines, x, y, w, h, fontSize = 10) {
  slide.addText(lines.join("\n"), {
    x,
    y,
    w,
    h,
    fontFace: "Avenir Next",
    fontSize,
    breakLine: false,
    color: COLORS.ink,
    margin: 0,
    valign: "top",
    fit: "shrink",
  });
}

/**
 * Footer bars carry the shared production contract so the viewer knows the
 * images came from the same canonical workflow rather than three unrelated
 * experiments.
 */
function addFooterBar(slide, text) {
  slide.addShape("rect", {
    x: 0.5,
    y: 6.82,
    w: 12.33,
    h: 0.42,
    line: { color: COLORS.panelEdge, pt: 0.8 },
    fill: { color: "EFE6D8" },
  });

  slide.addText(text, {
    x: 0.68,
    y: 6.94,
    w: 11.95,
    h: 0.14,
    fontFace: "Avenir Next",
    fontSize: 8.5,
    color: COLORS.muted,
    margin: 0,
    align: "center",
  });
}

/**
 * Slide 1: use the new posterior distribution corner plots the user generated
 * today. The purpose is to anchor the audience in the actual posterior shapes
 * before moving into predictive validation.
 */
function buildCornerSlide(pptx, profileData) {
  const slide = pptx.addSlide();
  addSlideChrome(
    slide,
    "Posterior Corner Plots",
    "Latest production posterior corners created on 2026-03-11; these runs also feed the canonical PPC and posterior-trend summaries.",
    "Slide 1 / 3"
  );

  const left = { x: 0.5, y: 1.25, w: 6.0, h: 5.35 };
  const right = { x: 6.83, y: 1.25, w: 6.0, h: 5.35 };

  addProfileCard(slide, profileData.devauc, left);
  addProfileCard(slide, profileData.sersic, right);

  slide.addImage({
    path: profileData.devauc.cornerImage,
    ...imageSizingContain(
      profileData.devauc.cornerImage,
      left.x + 0.1,
      left.y + 0.46,
      left.w - 0.2,
      4.52
    ),
  });
  slide.addImage({
    path: profileData.sersic.cornerImage,
    ...imageSizingContain(
      profileData.sersic.cornerImage,
      right.x + 0.1,
      right.y + 0.46,
      right.w - 0.2,
      4.52
    ),
  });

  slide.addText("Profile-specific model note: n is fixed to 4 in the de Vaucouleurs branch.", {
    x: left.x + 0.2,
    y: left.y + 5.0,
    w: left.w - 0.4,
    h: 0.34,
    fontFace: "Avenir Next",
    fontSize: 9.2,
    color: COLORS.muted,
    margin: 0,
    align: "center",
  });

  slide.addText("Profile-specific model note: Sersic carries n-dependent structure in the full workflow.", {
    x: right.x + 0.2,
    y: right.y + 5.0,
    w: right.w - 0.4,
    h: 0.34,
    fontFace: "Avenir Next",
    fontSize: 9.2,
    color: COLORS.muted,
    margin: 0,
    align: "center",
  });

  addFooterBar(
    slide,
    "Shared 12D hyper-parameter family; profile difference enters through the light-profile treatment. These exact runs are the sources for the canonical PPC and Fig. 8-like summaries shown next."
  );

  warnIfSlideHasOverlaps(slide, pptx);
  warnIfSlideElementsOutOfBounds(slide, pptx);
}

/**
 * Slide 2: canonical posterior predictive histograms.
 * We keep the images large and reserve the lower card area for the three most
 * important takeaways from `ppc_summary.json`.
 */
function buildPpcSlide(pptx, profileData, ppcSummaries) {
  const slide = pptx.addSlide();
  addSlideChrome(
    slide,
    "Canonical Posterior Predictive Check",
    "2026-03-10 canonical rerun: tail-capped full chain, burn-in 2000, 192000 posterior draws, candidate pool 100000, process-pool workers 12.",
    "Slide 2 / 3"
  );

  const left = { x: 0.5, y: 1.25, w: 6.0, h: 5.35 };
  const right = { x: 6.83, y: 1.25, w: 6.0, h: 5.35 };

  addProfileCard(slide, profileData.devauc, left);
  addProfileCard(slide, profileData.sersic, right);

  slide.addImage({
    path: profileData.devauc.ppcImage,
    ...imageSizingContain(
      profileData.devauc.ppcImage,
      left.x + 0.12,
      left.y + 0.5,
      left.w - 0.24,
      2.95
    ),
  });
  slide.addImage({
    path: profileData.sersic.ppcImage,
    ...imageSizingContain(
      profileData.sersic.ppcImage,
      right.x + 0.12,
      right.y + 0.5,
      right.w - 0.24,
      2.95
    ),
  });

  slide.addShape("rect", {
    x: left.x + 0.15,
    y: left.y + 3.66,
    w: left.w - 0.3,
    h: 1.42,
    line: { color: COLORS.panelEdge, pt: 0.7 },
    fill: { color: "FBF7F0" },
  });
  slide.addShape("rect", {
    x: right.x + 0.15,
    y: right.y + 3.66,
    w: right.w - 0.3,
    h: 1.42,
    line: { color: COLORS.panelEdge, pt: 0.7 },
    fill: { color: "FBF7F0" },
  });

  addSummaryLines(
    slide,
    buildPpcHighlights(ppcSummaries.devauc),
    left.x + 0.28,
    left.y + 3.83,
    left.w - 0.56,
    1.06,
    9.3
  );
  addSummaryLines(
    slide,
    buildPpcHighlights(ppcSummaries.sersic),
    right.x + 0.28,
    right.y + 3.83,
    right.w - 0.56,
    1.06,
    9.3
  );

  addFooterBar(
    slide,
    "Observed medians land near the bulk of the replicated distributions for both profiles. The most persistent tension is the unusually small observed sigma scatter relative to the replicated sigma std panel."
  );

  warnIfSlideHasOverlaps(slide, pptx);
  warnIfSlideElementsOutOfBounds(slide, pptx);
}

/**
 * Slide 3: Fig. 8-like trends.
 * The images are tall, so each half-slide reserves a narrow text column on the
 * right instead of pushing all commentary to the bottom. This keeps the trend
 * panels readable while still surfacing the quantitative selection offsets.
 */
function buildFig8Slide(pptx, profileData, fig8Summaries) {
  const slide = pptx.addSlide();
  addSlideChrome(
    slide,
    "Fig. 8-like Posterior Trends",
    "Separate evaluator from histogram PPC: fixed mass-grid conditional means for m5, gamma, and sigma_ap with parent / detectable / SLACS-like selected populations.",
    "Slide 3 / 3"
  );

  const left = { x: 0.5, y: 1.25, w: 6.0, h: 5.35 };
  const right = { x: 6.83, y: 1.25, w: 6.0, h: 5.35 };

  addProfileCard(slide, profileData.devauc, left);
  addProfileCard(slide, profileData.sersic, right);

  slide.addImage({
    path: profileData.devauc.fig8Image,
    ...imageSizingContain(
      profileData.devauc.fig8Image,
      left.x + 0.15,
      left.y + 0.52,
      3.38,
      4.56
    ),
  });
  slide.addImage({
    path: profileData.sersic.fig8Image,
    ...imageSizingContain(
      profileData.sersic.fig8Image,
      right.x + 0.15,
      right.y + 0.52,
      3.38,
      4.56
    ),
  });

  slide.addShape("rect", {
    x: left.x + 3.7,
    y: left.y + 0.52,
    w: 2.12,
    h: 4.56,
    line: { color: COLORS.panelEdge, pt: 0.7 },
    fill: { color: "FBF7F0" },
  });
  slide.addShape("rect", {
    x: right.x + 3.7,
    y: right.y + 0.52,
    w: 2.12,
    h: 4.56,
    line: { color: COLORS.panelEdge, pt: 0.7 },
    fill: { color: "FBF7F0" },
  });

  addSummaryLines(
    slide,
    buildFig8Highlights(fig8Summaries.devauc),
    left.x + 3.86,
    left.y + 0.72,
    1.8,
    4.1,
    8.9
  );
  addSummaryLines(
    slide,
    buildFig8Highlights(fig8Summaries.sersic),
    right.x + 3.86,
    right.y + 0.72,
    1.8,
    4.1,
    8.9
  );

  addFooterBar(
    slide,
    "Current production trend contract: n_posterior_draws = 256, n_mass_grid = 40, logM* in [11.2, 12.1], n_candidate_per_mass = 2000. Both profiles show rising m5 and falling gamma with mass; selection boosts sigma_ap in both cases."
  );

  warnIfSlideHasOverlaps(slide, pptx);
  warnIfSlideElementsOutOfBounds(slide, pptx);
}

/**
 * Main entrypoint:
 * - validate all required artifacts before any slide is created
 * - build slides in a deterministic order
 * - write a PPTX that is ready for render/overflow checks
 */
async function main() {
  Object.values(PROFILE_RUNS).forEach((profile) => {
    [
      profile.cornerImage,
      profile.ppcImage,
      profile.ppcSummary,
      profile.fig8Image,
      profile.fig8Summary,
    ].forEach(assertExists);
  });

  const ppcSummaries = {
    devauc: readJson(PROFILE_RUNS.devauc.ppcSummary),
    sersic: readJson(PROFILE_RUNS.sersic.ppcSummary),
  };
  const fig8Summaries = {
    devauc: readJson(PROFILE_RUNS.devauc.fig8Summary),
    sersic: readJson(PROFILE_RUNS.sersic.fig8Summary),
  };

  const pptx = new PptxGenJS();
  pptx.layout = "LAYOUT_WIDE";
  pptx.author = "OpenAI Codex";
  pptx.company = "CMASS Lens Project";
  pptx.subject = "Posterior corners, canonical PPC, and Fig. 8-like trend summary";
  pptx.title = "CMASS lens profile summary";
  pptx.lang = "en-US";
  pptx.theme = {
    headFontFace: "Avenir Next",
    bodyFontFace: "Avenir Next",
    lang: "en-US",
  };

  buildCornerSlide(pptx, PROFILE_RUNS);
  buildPpcSlide(pptx, PROFILE_RUNS, ppcSummaries);
  buildFig8Slide(pptx, PROFILE_RUNS, fig8Summaries);

  await pptx.writeFile({ fileName: OUTPUT_PPTX });
  console.log(`Wrote deck to ${OUTPUT_PPTX}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
