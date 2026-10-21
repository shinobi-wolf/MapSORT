# Review comments on the MapSORT paper

Comments are appended per draft iteration. Each item notes how it was handled in the next draft.

## Iteration 1 (2026-10-17, on draft 1)

### I. Global
- Don't use the "--" dashes.
  *Draft 2: removed from all prose, captions, tables and figure labels. Page ranges in the reference list are still typeset by the bibliography style.*
- What is the deal with the huge margins, especially at the bottom of the pages?
  *Draft 2: the template's single-column trim (a 131 x 194 mm text block on A4) is replaced by 25 mm margins all round.*

### II. Title
- Use "MapSORT: Landmark Association for Identity-Persistent Tracking of Stationary Crops and Fruit". Get rid of "map-frame".
  *Draft 2: done.*

### 1. Introduction
- The keywords should come right after the abstract.
- The first page after the abstract should start with the introduction. A figure sits at the top there, and it should not be placed there.
- Figure 1 is not very motivating and doesn't show a gap in the current SOTA. Figure 1 should be a teaser of what's to come: a comparison of two images, with BoT-SORT failing on the left/top and MapSORT succeeding in the same scenario on the right/bottom.
  *Draft 2: Fig. 1 is now a BoT-SORT vs. MapSORT teaser placeholder (no BoT-SORT run exists yet), placed below the opening paragraphs so that page 2 starts with text. The keywords now follow the abstract directly.*

### 3. Materials and methods
- 3.1: remove the sentence "Tracking is offline: the assignment may use the complete video, which is the setting of yield estimation and phenotyping from recorded passes".
  *Draft 2: done.*
- 3.2: remove mentions of "offline" everywhere in the paper, unless specifically asked for somewhere.
  *Draft 2: "offline" now appears only in the Conclusion.*
- 3.8: along with the four public third-party datasets, include our own dataset collected at IISc. A camera (RealSense D435i) mounted on a DJI Matrice 600 hexacopter was flown around three trees in a mango orchard in Bengaluru. It is not an MOT dataset: there is only a GT yield count per tree, so it is used only to evaluate the yield count.
  *Draft 2: added as "MangoYield", with a dedicated yield-count table (Sect. 4.4). The name is a placeholder. The text assumes one flight per tree; confirm this, along with the frame counts, resolution and how the GT counts were obtained.*
- 3.9: YOLO will be trained soon, so don't say the detections are derived from GT. Rewrite 3.9 as if the YOLO detectors were already in use.
  *Draft 2: done. YOLO version, input size, confidence threshold, training data per dataset and AP50 are TBD.*
- 3.10: the ReID model will also be trained per dataset. Write it up as if that were already done.
  *Draft 2: done. OSNet-x0.25 is initialised from MSMT17 and fine-tuned per dataset; the fine-tuning details are TBD.*

### 4. Results
- Never create two separate sets of results, with and without YOLO. Strip Sects. 4.1, 4.2, 4.3, 4.4, 4.5, 4.7 and 4.8 of any mention that GT detections are used, and remove Sect. 4.6. Keep the values currently in the tables and figures; they will be replaced with YOLO results later.
  *Draft 2: done.*
- 4.7: remove the "offline" mentions.
  *Draft 2: done.*

### 5. Discussion
- This section needs to be expanded with discussion of the other datasets, once all evaluations are finished.
  *Draft 3: done. One paragraph per viewing pattern (LettuceMOTS, MOrangeT, GrapeMOTS, AppleMOTS), plus consequences and limitations.*

### 6. Conclusion
- This is the only place where the paper mentions that MapSORT is offline. Treat the offline aspect as a fleeting limitation: recent online mappers show promise, and a future extension of this work will target it.
  *Draft 2: done, citing CUT3R and MASt3R-SLAM.*

### Appendix A
- The table should come after the section heading.
  *Draft 2: the table is pinned below the heading.*

## Draft 3 (2026-10-21): full revision, no new comments

The method changed, so the paper is rewritten to it. The Iteration 1 rules still hold:
- no "--" dashes;
- "offline" appears only in the Conclusion;
- there is one results set, written as YOLO plus fine-tuned ReID;
- the keywords follow the abstract, and page 2 starts with text.

- Method: MapSORT is now BoT-SORT plus a per-frame camera, fitted to the tracks' own landmarks (DLT or plane homography). The camera drives two stages: map motion, which re-matches unmatched tracks, and revival of lost and dormant tracks by reprojection. Revival is edge-aware, and new tracks at the frame edge wait for it. Sect. 3, Fig. 2 and Alg. 1 are rewritten; the StrongSORT cascade is gone.
- Baselines: StrongSORT is removed as a baseline; BoT-SORT and SAM2MOT remain.
- Fig. 1: the teaser placeholder is replaced by a real BoT-SORT vs. MapSORT frame (LettuceMOTS 0011), which the qualitative section also uses.
- Results: all tables and figures come from runs eval-20261021-013935 (baselines) and eval-20261021-191525 (MapSORT), on all test sequences.
- Appendix B is new. It gives per-sequence results for all 21 test sequences, backing the per-sequence comparison with BoT-SORT.
- Statistics: the dataset and per-sequence statistics are filled in, and the scene-map table is recomputed from the current reconstructions.
- Still TBD:
  - detector details and AP50;
  - ReID fine-tuning data;
  - SfM times;
  - the AppleMOTS and MangoYield landmark filters;
  - all MangoYield values;
  - the three qualitative panels of Fig. 7;
  - affiliation, acknowledgements and funding.
