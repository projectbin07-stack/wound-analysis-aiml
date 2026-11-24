# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.1] - 2026-11-24

### Fixed

- **Fallback mode reported a confidence it did not have.** With no model loaded,
  the heuristic mask was passed in as its own "probability map", so every
  fallback result reported 100 % confidence, and a heuristic score, severity
  band, "infection" band and "danger" band were derived from it. A plain grey
  image with a faint shadow came back as area 6.25 %, confidence 100 %, Mild,
  infection Medium. Fallback mode now returns no probability map, and
  `confidence`, `risk`, `health_score`, `severity`, `redness_band` and
  `healthy_pct` are `null`, with `"scored": false`. The text summary says no
  model is loaded and that the outline is not a wound detection.
- The fallback no longer outlines a region in a featureless image: if the
  centre's grey-level standard deviation is below 8 the mask is empty.
- The redness band started at a ratio of 0.2, so any neutral grey, white or
  black region (R/G of about 1) was put in "Medium", in model mode too. It now
  starts at 1.0 (range 1.0-3.0). In model mode this lowers the redness
  component, and so the heuristic score, for every image whose mask is not
  clearly red.

### Changed

- Removed medical wording. The `infection` field is renamed `redness_band`
  (it measures colour, not infection), the `danger` field is removed, and the
  summary template and LLM prompt no longer recommend actions such as consulting
  a professional; they state that the values are image measurements and not a
  diagnosis. Dashboard labels: "Risk assessment" -> "Heuristic score",
  "Infection risk" -> "Redness band", "Wound area" -> "Mask area",
  "Healthy %" -> "Outside mask %", "Confidence" -> "Mean model probability";
  scored fields show "n/a (no model)" in fallback mode.
- These are API changes (`infection` and `danger` removed, `redness_band` and
  `scored` added, fields may be `null`).

## [0.1.0] - 2026-11-18

First tagged release of the research prototype.

[Unreleased]: https://github.com/projectbin07-stack/wound-analysis-aiml/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/projectbin07-stack/wound-analysis-aiml/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/projectbin07-stack/wound-analysis-aiml/releases/tag/v0.1.0
