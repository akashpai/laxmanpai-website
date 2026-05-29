# Intraoral Scan from Smartphone Video (research prototype)

Convert a smartphone video of teeth (taken through a retractor/macro
"grinscope"-like device, which only sees **buccal/facial** surfaces) into a
metrically-scaled intraoral 3D model — and **predict the unseen lingual/occlusal
surfaces** with a statistical shape prior, with honest per-region accuracy and
per-vertex uncertainty.

> **Read first:** [`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md) (cited
> state-of-the-art + feasibility verdict) and [`docs/ROADMAP.md`](docs/ROADMAP.md)
> (phased implementation plan).

## The core idea in one picture

```
 smartphone video ──▶ sharp keyframes ──▶ SfM/MVS or 3DGS ──▶ metric scale
 (buccal only)         (video/frames)      (reconstruction)    (calibration)
                                                                    │
                                              buccal point cloud (MEASURED)
                                                                    │
                          segmentation (FDI + buccal/lingual/occlusal)
                                                                    │
                              Statistical Shape Model partial fit
                                       (completion/ssm)
                                                                    ▼
                       full arch = buccal (measured) + lingual/occlusal (PREDICTED,
                                    with per-vertex uncertainty)
                                                                    │
                                      validation vs ground-truth IOS
                                       (validation/metrics, per region)
```

## The honest verdict

- **Visible buccal surfaces:** ~0.5 mm is achievable **if** a known-size metric
  reference is in the camera's focal plane. Without that, scale is undefined.
- **Hidden lingual/occlusal surfaces:** these are **predicted, not measured**.
  Realistic accuracy floor **~0.1–0.25 mm** (molar occlusal worst). Good for
  visualization / orthodontic monitoring / aligner arch-form (<0.25 mm); **not**
  for restoration margins (≤0.12 mm) or implants (~0.01 mm).
- You **cannot** recover roots, intaglio, or sulcus from video alone — that needs
  CBCT or X-ray.

See the report for the literature behind every number.

## Install & test

```bash
cd intraoral-scan
pip install -r requirements.txt          # core: numpy, scipy (+ optional cv2/trimesh)
pytest -q                                # 15 tests, runs on numpy+scipy alone
python scripts/demo_completion.py        # synthetic end-to-end demo
```

The demo prints the key asymmetry you should expect on real data — buccal
(measured) error far below hidden (predicted) error:

```
buccal : mean≈0.02 mm      (measured)
hidden : mean≈0.28 mm      (predicted from the prior)
```

## Layout

| Path | What it does | Status |
|---|---|---|
| `completion/ssm.py` | PCA shape model: train, fit-to-partial, per-vertex uncertainty | ✅ tested |
| `validation/metrics.py` | ICP best-fit, surface deviation, trueness/precision | ✅ tested |
| `calibration/scale.py` | metric scale (reference distance / multi-bar / ArUco) | ✅ |
| `video/frames.py` | sharp keyframe extraction (variance-of-Laplacian) | ✅ |
| `reconstruction/colmap_runner.py` | COLMAP SfM+MVS wrapper | ✅ (needs COLMAP) |
| `segmentation/interface.py` | pluggable FDI segmenter interface | ✅ |
| `pipeline.py` | orchestration + per-region validation | ✅ tested |
| `scripts/demo_completion.py` | synthetic end-to-end demo | ✅ |

External engines (COLMAP, 3D Gaussian Splatting, deep segmenters) are wrapped
behind clean interfaces and are **not** required to run the tested core.

## Status

Early research prototype. The reconstruction/segmentation backbones are wrappers/
interfaces to be filled in (Phases 2–3 of the roadmap); the **scientifically
load-bearing parts — SSM completion of unseen surfaces, metric-scale recovery,
and the validation framework — are implemented and tested.** Not a medical
device; not for clinical use.
