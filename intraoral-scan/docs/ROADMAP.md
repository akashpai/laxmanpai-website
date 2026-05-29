# Implementation Roadmap

A phased, buildable plan tailored to your goal (**clinical product for
clear-aligner / restoration manufacturing**) and your stated assets (**you
already have some paired data**). Read alongside `RESEARCH_REPORT.md` for the
evidence behind every accuracy figure and feasibility claim.

> **Reframe the spec first.** "0.5 mm to ground truth, everywhere" is not a
> single number you can promise. Re-state the target as a **per-region spec**:
> - Buccal (measured): target ≤ 0.10–0.20 mm.
> - Lingual (predicted): target ≤ 0.25 mm, *reported with uncertainty*.
> - Occlusal molar (predicted): best-effort, expect 0.25–0.40 mm.
> - Roots / sulcus: out of scope without CBCT/X-ray.
> This is the difference between a credible medical-device program and a claim
> that will fail validation.

---

## Phase 0 — Spec, ethics, and a falsifiable accuracy contract (2–4 weeks)
- Lock the **per-region accuracy spec** above; define pass/fail per intended use
  (aligner arch-form vs restoration margins — they are different products).
- Decide the **intended use claim** narrowly. The defensible v1 claim is
  *"visualization / orthodontic monitoring / aligner arch-form,"* **not**
  *"fabrication-grade full-arch replacement for an IOS."*
- IRB/ethics + data-governance plan for patient scans (this is PHI).
- Define the **validation protocol now** (Phase 5) so you build toward it.

## Phase 1 — Capture rig + metric scale (the make-or-break) (4–8 weeks)
- Build the retractor/"grinscope" with an **integrated, precisely-known ChArUco
  target in the teeth focal plane** (see `calibration/scale.py`). Without a
  metric reference, every accuracy number is meaningless.
- Standardize a **capture protocol**: lighting (cross-polarization to kill enamel
  specular highlights), drying, slow sweep covering all visible buccal surfaces,
  frame rate, distance.
- Implement frame selection (`video/frames.py`, variance-of-Laplacian) — done.
- **Deliverable:** a repeatable capture that yields sharp, scale-referenced
  keyframes. Validate scale recovery against a caliper-measured object to
  ≤0.1 mm.

## Phase 2 — Visible-surface reconstruction (6–12 weeks)
- Baseline: **COLMAP SfM+MVS** (`reconstruction/colmap_runner.py`) — done as a
  wrapper. Inject the ChArUco corners for metric bundle adjustment.
- Specular fallback: **3D Gaussian Splatting / NeuS** (cf. Dental3R, DentalSplat)
  for the wet/low-texture regime.
- **Gate:** validate *buccal-only* reconstruction against ground-truth IOS
  (restricted to buccal vertices) using `validation/metrics.py`. Target mean
  ≤ 0.15 mm before proceeding. If you can't hit this on visible surfaces, the
  hidden-surface problem is moot.

## Phase 3 — Segmentation + correspondence (6–10 weeks)
- Train/adopt an IOS tooth segmenter (iMeshSegNet/TeethGNN/transformer) on
  **Teeth3DS** → per-vertex **FDI** label + buccal/lingual/occlusal class
  (`segmentation/interface.py`). This is a solved problem (Dice ~0.96–0.98).
- Establish **dense correspondence** to a per-tooth template (the prerequisite
  for the SSM). This is the real engineering work: non-rigid registration of each
  segmented tooth to a canonical template so vertex *i* means the same anatomical
  point across all scans.

## Phase 4 — Hidden-surface prior (the core IP) (10–16 weeks)
- Build per-tooth-type **Statistical Shape Models** from corresponded Teeth3DS
  meshes (`completion/ssm.py`) — implemented and tested. Optionally upgrade to a
  learned completion net (point-to-mesh / diffusion) once the SSM baseline is
  measured.
- Fit the SSM to the buccal reconstruction; predict lingual/occlusal vertices
  with **per-vertex uncertainty** (`pipeline.complete_hidden_surfaces`).
- **Use contralateral mirroring** as an auxiliary prior where the opposite tooth
  was better captured (note: imperfect — asymmetry is real, see report).
- **Honest-uncertainty calibration:** the SSM's internal `per_vertex_std`
  *under-reports* error when patient anatomy lies outside the model span (the
  demo shows this). **Calibrate uncertainty empirically** on held-out paired data
  (conformal prediction / reliability curves), don't trust the model variance.

## Phase 5 — Validation (continuous; formal at v1) (ongoing)
- Use the trueness/precision + per-region surface-deviation framework in
  `validation/metrics.py` (ICP best-fit, mean/RMS/p90/p95/Hausdorff).
- **Report per region and per tooth type**, never a single global number.
- Statistical validation of *predicted* surfaces: hold-out cases where the hidden
  region is independently measured (CBCT/sectioned models); compare predicted-vs-
  measured RMS against thresholds. Per the report, inferred geometry can only be
  validated population-level, never per-patient.
- Track against the bars: aligner <250 µm; restoration ≤120 µm; IOS benchmark
  ~20–115 µm.

## Phase 6 — Regulatory & productization (parallel from Phase 0)
- **FDA:** Class II 510(k) if a predicate fits (DentalMonitoring/IOS/aligner
  software); **De Novo** if not. **EU MDR:** ≥ Class IIa SaMD (Rule 11).
- Predetermined Change Control Plan (PCCP) for the ML components.
- Ground truth per FDA AI guidance (≥3 clinical experts); subgroup/
  generalizability analysis.
- **Restoration/implant claims are the hardest** — likely require captured
  (not predicted) hidden surfaces, i.e. fuse a low-dose CBCT or an actual IOS for
  the posterior/lingual. Consider a **hybrid product**: phone video for anterior
  buccal + monitoring, real IOS/CBCT where margins matter.

---

## Data strategy (you have *some* paired data — leverage it)
1. **Bootstrap the prior from public IOS** (Teeth3DS) via correspondence + SSM —
   no paired data needed. *(Verify Teeth3DS license before any product use.)*
2. **Synthetic paired data:** render buccal "photos/video" from IOS meshes
   (as TeethDreamer does) to pretrain the photo→3D mapping at scale.
3. **Your real paired data** (phone video + IOS/CBCT) is gold — reserve it for
   (a) domain-adaptation fine-tuning and (b) the **held-out validation set**.
   Never train on your validation cases.
4. Plan ongoing paired collection through a clinic partner; power it for your
   per-region accuracy claims.

## What's already in this repo (runnable today)
- `completion/ssm.py` — SSM training + partial fit + per-vertex uncertainty ✅ tested
- `validation/metrics.py` — ICP, surface deviation, trueness/precision ✅ tested
- `calibration/scale.py` — metric scale (reference distance / multi-bar / ArUco) ✅
- `video/frames.py` — sharp keyframe extraction ✅ (scoring tested)
- `reconstruction/colmap_runner.py` — COLMAP SfM+MVS wrapper ✅
- `segmentation/interface.py` — pluggable segmenter interface ✅
- `pipeline.py` — orchestration + per-region validation ✅
- `scripts/demo_completion.py` — end-to-end synthetic demo ✅ runs

## Biggest risks (ranked)
1. **Metric scale** from handheld video at sub-mm — unsolved in the wild; the
   ChArUco-in-focal-plane rig is the mitigation.
2. **Hidden molar occlusal anatomy** — high-frequency, patient-specific; the
   prior cannot recover it. Mitigate by *scoping it out* of fabrication claims.
3. **Regulatory acceptance of predicted geometry** — validate population-level;
   keep margin-critical surfaces measured, not inferred.
4. **Specular enamel** breaking reconstruction — cross-polarization + 3DGS.
5. **Dataset licensing** (Teeth3DS) and PHI governance.
