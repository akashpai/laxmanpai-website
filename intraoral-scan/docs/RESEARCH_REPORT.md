# Smartphone Video → Intraoral Scan: State of the Art & Feasibility

**Scope.** Convert a smartphone video of teeth (captured via a retractor/macro
"grinscope"-like device, showing only **buccal/facial** surfaces) into a
clinical-grade, metrically accurate **full-arch intraoral 3D scan** (target
~0.5 mm vs ground truth, ideally tens of microns) for **clear-aligner and
restoration manufacturing**.

**Method.** Five parallel research agents fanned out across web search + academic
sources (PubMed/PMC, arXiv, Hugging Face Papers). Numbers below are quoted as
published with sources inline. Verification caveat: several primary hosts
(PMC, arXiv HTML, ScienceDirect, nature.com) returned HTTP 403 to direct
fetching, so some exact figures come from search snippets/abstracts and should
be re-verified against the primary PDFs before any regulatory or publication use.
Confidence is flagged per claim where relevant.

---

## TL;DR — the honest feasibility verdict

| Sub-problem | Achievable at ~0.5 mm? | At clinical ~0.05 mm? |
|---|---|---|
| **Visible buccal crown surfaces** (with a metric reference in frame) | **Yes** | Borderline; not yet demonstrated from handheld *video* |
| **Lingual/palatal surfaces** (predicted) | **Maybe** (~0.1–0.25 mm floor) | **No** |
| **Molar occlusal anatomy** (predicted) | **Hardest case**, often >0.25 mm | **No** |
| **Tooth roots / intaglio / sulcus** (predicted, no extra modality) | **No** (needs CBCT/X-ray) | **No** |

**Bottom line.** The visible-surface half is an engineering problem that is
largely solvable to ~0.5 mm with a proper metric reference. The unseen-surface
half is **fundamentally bounded by a statistical prior**: you cannot measure what
the camera never saw, so lingual/occlusal geometry is a *prediction*, with a
realistic accuracy floor of **~0.1–0.25 mm** — adequate for orthodontic
visualization/monitoring and possibly aligner arch-form (tolerance <0.25 mm),
but **not** for restoration margins (≤0.12 mm) or implants (~0.01 mm). No
published system today produces a clinical-grade full-arch model (crown +
sulcus, tens of microns) from ordinary smartphone video.

---

## 1. Reconstructing the VISIBLE (buccal) surfaces from video

**Photogrammetry (SfM + MVS) on dental casts is already near 0.5 mm.**
- Smartphone stereophotogrammetry of orthodontic casts: mean deviation **0.55 mm
  (Agisoft Metashape) / 0.50 mm (3DF Zephyr)** vs CBCT; repeatability 0.01–0.03 mm.
  Errors concentrate in **interproximal areas and occlusal cusps/fissures**.
  *J Orthod Sci / ScienceDirect S2212443822000686, 2022.*
- Photogrammetry vs lab scanner on casts: **RMS 0.16–0.37 mm** (proof of concept;
  exact scale method uncertain — verify). *ResearchGate 333073071.*

**Intraoral-specific neural pipelines (2025) confirm standard SfM is fragile on
teeth and are moving to Gaussian Splatting.**
- **Dental3R** (arXiv:2511.14315, 2025): wavelet-regularized 3D Gaussian
  Splatting, bypasses brittle SfM; reports **0.949 SSIM, 0.18 mm ASSD**. Notes
  "large view baselines, inconsistent illumination, and specular enamel …
  destabilize camera recovery."
- **DentalSplat** (arXiv:2511.03099, 2025): 3DGS from sparse intraoral photos —
  but targets novel-view synthesis, not certified metric accuracy.
- **TeethDreamer** (MICCAI 2024, arXiv:2407.11419): 3D teeth from **5 intraoral
  photos**, **Chamfer 0.167 mm, Hausdorff 2.11 mm** — but uses a **diffusion
  prior to hallucinate novel views**, and scale is fit against ground-truth IOS,
  not recovered from images. Dice 0.767 over 95 cases.

**Plain NeRF/3DGS fail on specular, translucent enamel.** Specular radiance
entangles with geometry (shape ambiguity), documented across Ref-NeuS
(arXiv:2303.10840), UniSDF (arXiv:2312.13285), GNeRP (arXiv:2403.11899). Enamel
also exhibits subsurface/volume scatter and is semi-translucent, degrading
structured-light and feature S/N (3Shape technical material; J Prosthet Dent
S1991790223000910 shows translucency measurably degrades scan accuracy).

**Metric scale MUST be injected (the make-or-break detail).**
- Monocular SfM/MVS is **scale-ambiguous**; metric scale needs an external
  reference: ArUco/coded targets, calibrated **scale bars** (best practice ≥3),
  or a known distance. (Agisoft docs; geodetic.com.)
- Calibrated photogrammetry scale bars are individually calibrated to **~0.1 mm
  or better**, bounding achievable metric accuracy.
- The reference **must sit in the same focal plane** as the teeth (very shallow
  intraoral macro depth of field), or scale is biased.
- Phone depth sensors are too coarse: **iPhone/iPad TrueDepth ~2 mm** accuracy
  (vs CBCT facial scan, mean discrepancy **0.387 ± 0.361 mm**); **LiDAR depth map
  only ~256×192 (~49k pts/frame)** — room/object scale, not tooth morphology.

**Confidence:** High that ~0.5 mm is reachable for visible crowns *with* a metric
reference. **Open gap:** no located study demonstrates validated sub-millimeter,
metrically-scaled buccal reconstruction from handheld monocular *video* (vs posed
photos or casts).

---

## 2. Predicting the UNSEEN surfaces (lingual, occlusal, roots) — the crux

**Statistical Shape Models can infer hidden geometry, with error growing as the
unseen region grows.**
- Classic dental SSM surface reconstruction: **Buchaillard et al., Comput Biol
  Med 37 (2007)** — reconstruct missing surface regions by fitting a prior;
  error grows with missing-region size.
- Root-from-crown SSM (feasibility, 71 datasets): predicted tooth-axis angle
  error **7.5±4.3° (upper) / 6.7±3.8° (lower)**; horizontal axis deviation
  **1.3±0.8 mm near the CEJ, 0.7±0.5 mm at the apical third**. *Int J CARS 2022,
  PMC9468133.*
- Crown→root form prediction shows **no significant population-level difference**
  per tooth type (mandibular 1st premolar SSM from 76 extracted teeth). *Durschlag,
  Loma Linda thesis.*

**Generative occlusal/crown completion — good on small patches, breaks at
boundaries.**
- Molar occlusal GAN reconstruction (StyleGAN-2 + Bayesian), 92 molar meshes:
  RMSE **0.02 mm (small mask) → 0.16–0.18 mm at ~80% removed**; error scales
  ~**linearly** with missing-area fraction. *J Dentistry 145 (2024), PMC11058210.*
- **Critical boundary finding:** identical 30% area removed gives **0.024 mm
  (centered) vs 0.15 mm (~6×) when the mask touches the outer contour.** The
  buccal-only case removes the entire occlusal/lingual boundary — predicting the
  worst regime. *Same paper, PMC11058210.*
- DentalRecNet (dual-discriminator GAN): occlusal RMS **0.114 mm**. *PMC9018184.*
- Point-to-mesh crown completion: Chamfer **≈0.062** (normalized). *Med Image
  Anal 2024, S1361841524003645.*
- Memory/retrieval-augmented point-cloud completion is an active 2025 line
  *because* dental clouds have large missing regions (arXiv:2512.03598).

**High accuracy on hidden surfaces only happens when the surface was actually
captured by another modality — a key counterexample.**
- DL "integrated tooth models" reach **0.02 mm (maxilla)/0.03 mm (mandible)** —
  *but only because real CBCT roots were used* (measured, not predicted).
  *Prog Orthod 2022, PMC9081076.*
- CBCT-to-IOS crown-root fusion: RMS **0.036–0.095 mm** (hidden surface imaged).
  *BMC Oral Health 2024, PMC11684253.*
- ToothInpaintor adds unseen roots **using a 2D panoramic X-ray** as extra
  evidence (arXiv:2211.15502).

**The fundamental limit (irreducible variance).**
- Occlusal/lingual anatomy is the **most individually variable** region: AI vs
  human crown designs differ significantly on **occlusal and distal** surfaces
  but **not** on mesial/margin (P>.05). *J Dentistry 2024, S0020653924001965.*
- State-of-the-art AI crown design floor: global RMSE **~80 µm median, max
  ~225 µm** — genuine population variance no prior removes. *S0109564123004062.*
- Contralateral mirror symmetry is an **imperfect** prior: significant left-right
  asymmetry in nearly all post-canine teeth; greatest in maxillary premolars and
  mandibular 1st molars. *PMC12033903, 2025.*

**Synthesis (falsifiable):** No published work reconstructs a *fully unseen*
lingual/occlusal surface from buccal-only smartphone video. Combining the
linear area–error law, the ~6× boundary penalty, and the ~0.08–0.23 mm AI-design
variance floor, the realistic accuracy floor for predicted occlusal/lingual
surfaces is **~0.1–0.25 mm at best** — 2–5× the ~0.05 mm clinical IOS bar.

---

## 3. Datasets & segmentation

- **Teeth3DS** (3DTeethSeg'22, MICCAI): **1,800 IOS meshes, 900 patients, 23,999
  teeth**, OBJ + per-vertex **FDI** labels (0 = gingiva), on OSF
  (osf.io/xctdy). Split 1,200/600. *arXiv:2305.18277.* **License not clearly
  stated — verify before clinical use (do not assume CC-BY).**
- **Teeth3DS+** extends it; **3DTeethLand (MICCAI 2024)**: first public 3D dental
  landmark set, 340 IOS. *arXiv:2512.08323.*
- **Paired smartphone-photo ↔ IOS public data effectively does not exist.** Best
  precedents are **private**: TeethDreamer (3,200 cases of 5 photos + IOS, not
  released); DenGaussDiff (~1,000 cases, not released). **Implication: build the
  photo→3D prior via synthetic rendering from IOS meshes, plus self-collected
  paired data.**
- CBCT (roots/full tooth): **CTooth/CTooth+** (22 + 146 volumes);
  **ToothFairy2** (530 CBCT, 480 public, **CC BY-SA**, 42 classes);
  **MMDental** (660 patients, Figshare); PhysioNet multimodal (329 CBCT).
- CBCT↔IOS registration: DDMF reached **0.17 mm** avg symmetric surface distance
  (private data). No large *public* paired CBCT↔IOS benchmark found.
- **Segmentation is essentially solved on clean IOS:** iMeshSegNet Dice **0.964**;
  recent methods Dice **0.98+**, 3DTeethSeg score 0.987. Families: MeshSegNet/
  iMeshSegNet, TSegNet, TeethGNN, point/transformer/Mamba (DilatedToothSegNet,
  T-Mamba, GeoT). So the binding constraint is **reconstruction + paired data**,
  not segmentation.
- Synthetic data: **TeethGenerator** (diffusion, arXiv:2507.04685),
  **ToothForge** (spectral, arXiv:2506.02702).

---

## 4. Existing systems & the accuracy gap

- **DentalMonitoring ScanBox Pro is a 2D photo-capture device, NOT a 3D
  scanner** — retractor + phone holder feeding AI monitoring; does not produce a
  fabrication-grade 3D model. A validation reported DM 3D models within
  **~0.001–0.028 mm mean *global* deviation** of an iTero reference, *within the
  ABO ±0.5 mm* bar — but that is averaged registered deviation against a loose
  orthodontic threshold, not per-tooth trueness. *AJODO 2019.* DM has **FDA De
  Novo + EU MDR Class IIa** as SaMD.
- **SmileDirectClub** (at-home impressions): ~30% vs ~5% rejection-rate
  discrepancy vs an independent orthodontist; DC AG settlement ($500k, 17k+ NDAs
  released); **Chapter 11 → shutdown Dec 2023 (~$900M debt)**; ADA FDA citizen
  petition; Hindenburg allegations. A cautionary tale for at-home/photo aligners.
- **Academic photo→3D:** TeethDreamer (5 photos, CD 0.167 mm); IEEE TVCG 2023
  parametric model **~1.01 mm² Chamfer, Dice 0.767**; 2025 J Dent Sci five-view
  (SAM + Depth Anything). **All reconstruct crown surfaces only — no roots, no
  intaglio/sulcus.**
- **Consumer phone photogrammetry** is far off clinically: KIRI significantly
  worse precision than IOS; phantom benchmark mean errors **2.9–21.4 mm** across
  apps. **Exception:** smartphone photogrammetry of **implant scan-body
  positions** (PIC app) can be clinically sufficient — but that measures
  high-contrast coded targets, not tooth/soft-tissue surfaces (an easier
  problem).
- **The benchmark to beat (real IOS):** in-vitro full-arch trueness/precision
  **~17–115 µm** (Primescan ~17–29 µm; Trios ~21–47 µm; iTero Lumina ~34 µm);
  in-vivo full-arch ~**55–90 µm**. Pooled means ~**76.6 µm trueness / 56.6 µm
  precision**. So smartphone reconstruction is currently **~2× (best, generative)
  to 10–100× (typical) worse** than a real intraoral scanner. *PMC7940805,
  PMC8303663.*

---

## 5. Validation & regulatory

- **Accuracy = trueness + precision (ISO 5725-1).** Trueness needs an
  independent reference (lab/industrial scanner, CBCT); precision = repeatability.
- **Standard workflow:** best-fit/ICP superimposition → signed nearest-neighbour
  deviation → **RMS + color map** (e.g. Geomagic Control X). **RMS is
  software-dependent** — only comparable within one tool. (PMC10756783;
  S0010482525001301.)
- **Standards gap:** **ISO 12836** (CAD/CAM digitizer test methods) **explicitly
  excludes hand-held scanners**; **ISO 1942** is dental vocabulary. There is **no
  ISO test-method standard purpose-built for hand-held IOS**.
- **Clinical thresholds:** restoration marginal gap acceptable **≤120 µm**
  (50–120 µm, McLean/von Fraunhofer); **implants ~10 µm**; **clear-aligner
  manufacturing accepted error <250 µm**. Meta-analytic IOS-vs-conventional
  full-arch difference **~152 µm** (high heterogeneity).
- **Regulatory:** IOS & aligners are **FDA Class II via 510(k)** (Invisalign
  aligner product code **NXC** 21 CFR 872.5470; planning software **PNN**).
  **De Novo** when no predicate (DentalMonitoring precedent). **EU MDR**:
  decision-support software ≥ **Class IIa** (Rule 11).
- **Can a *predicted* hidden surface pass validation?** FDA AI/ML guidance hinges
  on a defined **ground truth** (recommend **≥3 clinical experts**) and subgroup
  generalizability — which an *unobservable* surface lacks per-case. Validation of
  inferred geometry could only be **statistical/population-level** (hold-out cases
  where the hidden region was independently measured by CBCT/sectioning,
  comparing predicted-vs-measured RMS against the thresholds), **never
  per-patient**. Reinforcing skepticism: aligner **planned tooth movements are
  achieved only ~41–50%** (Kravitz 2009; Haouili 2020) — measured proof that
  *predicted* dental geometry diverges from reality. Net: inferred hidden geometry
  is plausibly acceptable for **aligner arch-form (250 µm)** but **not for
  margin-critical restorations (≤120 µm) or implants (~10 µm)**.

---

## Key load-bearing numbers (quick reference)

- Clinical IOS full-arch: **~20–115 µm** (target/benchmark).
- Restoration margin tolerance **≤120 µm**; implant **~10 µm**; aligner **<250 µm**.
- Best photo→3D (TeethDreamer, 5 photos, generative): **CD 0.167 mm**.
- Predicted hidden-surface realistic floor: **~0.1–0.25 mm**; molar occlusal worst.
- Reconstruction error scales ~linearly with missing area; **~6× penalty** when
  the missing region includes the boundary (the buccal-only regime).
- Teeth3DS: **1,800 IOS meshes / 900 patients** (license unverified).
- Public paired smartphone↔IOS data: **none found.**

---

## Caveats on verification
Some exact figures were extracted from search snippets/abstracts because PMC,
arXiv-HTML, ScienceDirect, and nature.com blocked direct fetching during
research. Before citing precise values in a regulatory submission or paper,
re-verify against the primary PDFs (especially: TeethDreamer metrics, the molar
GAN boundary/area numbers PMC11058210, root-from-crown PMC9468133, and the IOS
trueness aggregates).
