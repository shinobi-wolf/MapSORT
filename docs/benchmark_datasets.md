# Benchmark Datasets for MapSORT-Style Stationary-Object MOT with Revisits and 3D Landmarks

Only a few public datasets meet all three criteria (revisits, SfM-feasible imagery, stationary objects) and also ship persistent-ID tracking ground truth. The strongest are agricultural: **LettuceMOT** (explicit back-and-forth and out-and-in robot trajectories), **MOrangeT** (fruit re-entry, plus COLMAP reconstructions in the release) and **LFSD** (eight closed-loop RGB-D + RTK-GPS lettuce sequences). Outside agriculture, **Aria Digital Twin** and **KITTI-360** give the best stationary-instance GT with poses. Most other well-known agricultural datasets fail on revisits (single passes) or on GT (counts or still-image masks only).

## TL;DR
- **Best direct benchmarks (agricultural):** LettuceMOT, MOrangeT, LFSD and your own reference LettuceMOTS. All four are moving-camera video of stationary plants or fruit with persistent-ID MOT/MOTS GT and documented re-entry or loop closure. In MOrangeT, the authors' 3-D relocalization step is essential for long-term tracking: with ground-truth boxes, removing it drops HOTA from 0.935 to 0.434 (Santos et al., arXiv 2312.16724). This is close to a published ablation of the MapSORT idea.
- **Best non-agricultural benchmarks:** Aria Digital Twin (324 stationary object instances with 6DoF GT, egocentric revisits) and KITTI-360 (static objects carrying one instance ID across a whole driving sequence, with repeated routes). 3RScan is the best *cross-session* re-association testbed, but its objects can move between scans.
- **Near-misses worth adapting:** BUP20/BUP-ST20, PhenoBench, Sugar Beets 2016, CitrusFarm and EgoObjects have multi-session revisits but lack dense persistent-ID video GT. AppleMOTS, MinneApple, Fuji-SfM and the single-pass grape UAV set lack revisits or tracking GT. Revisit regimes can be built by concatenating sessions or reversing/splicing passes, but cross-session IDs would then need relabeling.

## Key Findings

1. **Revisits are rare and usually undocumented.** Most agricultural tracking datasets are recorded as one pass along one side of a row (MinneApple, the 2023 UAV grape set, very likely AppleMOTS).\[1\]\[2\] Only LettuceMOT, LFSD, MOrangeT, GrapeMOTS and LettuceMOTS explicitly document re-entry or looped trajectories.\[3\]\[4\]\[5\]
2. **Some 3D is already provided.** MOrangeT ships COLMAP results. Fuji-SfM ships camera positions, a point cloud and 3D apple GT. LFSD and BUP20 ship odometry/RTK and RGB-D.\[6\]\[7\] Aria Digital Twin, KITTI-360 and 3RScan ship full poses and geometry.\[8\]\[9\]\[10\] These let you separate tracker error from landmark-estimation error.
3. **Your own prior work already uses part of this stack.** Your LocalizeSORT manuscript (Drive: "LocalizeSORT_revision_editing.pdf") evaluated on LettuceMOT and LFSD sequence 3. On LettuceMOT B&F it reached HOTA 93.4 vs. 70.9 for StrongSORT, and on O&I 88.2 vs. 67.8.\[5\] MapSORT should be compared against that baseline on the same splits, plus MOrangeT and LettuceMOTS.
4. **There is no public moving-camera retail-shelf or warehouse MOT dataset with persistent product IDs.** I found none. Retail tracking datasets track shoppers from fixed cameras, which fails criteria 2 and 3.\[11\]\[12\]

## Details — Precision Agriculture / Horticulture

### A. Direct fits (revisits + video + persistent-ID GT)

| Dataset | Domain / objects | Capture | Size | Unique IDs | Revisits (how known) | 3D / SfM | GT format | Access | Baselines |
|---|---|---|---|---|---|---|---|---|---|
| **LettuceMOTS** (Hu et al., arXiv 2306.13518) – reference | Lettuce field, VegeBot, downward RGB | RGB video, 3 growth stages (Sep–Oct 2022) | 12 sequences, 1,308 images | 314 instances, 17,562 masks | **Documented**: method designed for re-identification "when the robot travels back and forth" | Not provided; feasible (downward overlapping video, quasi-planar ground) | KITTI-MOTS (RLE masks + object IDs) | GitHub NanH5837/LettuceMOTS; CC BY 4.0 per Voxel51 dataset card | Shape-feature MOTS: HOTA > 70, AssPr > 80, 29 FPS; beats two SOTA MOTS methods\[13\] |
| **LettuceMOT** (Hu et al., Front. Plant Sci. 2022) | Lettuce, rosette + heading stages | RGB, 130° lens at ~1.5 m, cropped to 810×1080, 10 FPS\[5\] | 8 sequences, 5,466 images | 707 tracks / 42,735 boxes (summed from Table 2) | **Documented**: B&F sequences reverse to the start; O&I sequences turn out of the row and come back. Re-occurrence is present only in these\[5\] | Not provided; RLoc/COLMAP already used by you | MOT format (DarkLabel) | MEGA link; article CC BY, no separate data license stated | Original paper: SORT, FairMOT, ByteTrack, ByteTrack+NSA (e.g., B&F1 ByteTrack+NSA MOTA 91.9 / HOTA 68.7 / IDF1 59.7); authors note IDSW is poor on B&F/O&I.\[14\] The LoFTR-based tracker of Hernandez & Medeiros (Comput. Electron. Agric. 226, 109379, 2024) reports that it "outperforms cutting-edge algorithms for robotic plant tracking in the LettuceMOT dataset"; LocalizeSORT (your draft) |
| **LFSD** (Wang et al., Front. Plant Sci. 2023) | Lettuce farm SLAM | D435i RGB-D, 1280×720 (seq 1–4) / 640×480 (seq 5–8), 10 FPS; IMU 200 Hz; RTK-GPS 1 Hz\[7\] | 8 closed-loop sequences, ~455–2,565 m each, ~67 GB\[15\] | Seq 1–4 annotated: 13,409 labeled frames, 4,406 tracks\[16\] | **Documented** loop closure ("Strict" for all 8): the robot drives multiple ridges and "returns to the first ridge". **Not documented**: whether a returning plant keeps its earlier ID. Seq 3's 2,468 tracks over 4 ridges suggest it may not (inference — check mot.txt)\[16\] | RTK + IMU + depth given; SLAM baselines (ORB-SLAM3, DSO, OA-SLAM)\[16\] | MOT format; frame field is the RGB timestamp\[15\]\[16\] | IEEE DataPort (doi 10.21227/s85q-cf10, possibly subscription) + GitHub wangshuo9707/LFSD-Dataset\[7\]\[17\]\[18\] | HOTA: ByteTrack+NSA 74.9–78.9, YOLOv5+SORT 64.3–67.5, FairMOT 39.2–51.9\[15\] |
| **MOrangeT** (Santos et al., Comput. Electron. Agric. 2024; arXiv 2312.16724) | Sweet orange, commercial groves, 4 varieties | Smartphone video in portrait, reported as 1940×1080 (9 of 12 videos on a Zhiyun Smooth 4 gimbal); ORB+FLANN frame selection "reduces the number of frames by 50 to 70 percent" (no constant fps) | 12 videos (one tree side each), 4,713 frames\[19\] | 1,198 tracks | **Documented**: Santos et al. note "the same orange enters and exits the field of view three times, entering at frames 53, 218 and 446" | **COLMAP SfM results included**; GT built by reprojecting SfM-modelled 3D spheres | MOT16 | Embrapa Redape, doi 10.48432/OI7BFG ("publicly available"; license not verified)\[20\] | YOLOv5l + IoU/Hungarian + 3D relocalization: HOTA 0.560, MOTA 0.585, count error 1.25%. Removing relocalization with GT boxes drops HOTA from 0.935 to 0.434\[21\] |
| **GrapeMOTS** (Ariza-Sentís et al., Data in Brief 2024) | Wine grape bunches (plus trunk/pole in 2 videos)\[22\] | DJI Phantom 4 RTK UAV RGB video\[23\] | 11 videos (8 "PathPlanning" multi-perspective, 3 non-planned)\[22\] | ~200–930 instances per video\[22\] | **Documented** multi-perspective capture; "The same ID was maintained for grape bunches seen from different perspectives"\[3\] | Not provided; UAV video + RTK makes SfM feasible | MOTS (per-pixel, CVAT)\[3\] | Zenodo 10625595 | Dataset paper only; GrapeOcclusionMOTS (Electronics 2026) adds occlusion labels and SAM-based tracker results\[24\] |

### B. Near-misses (fail ≥1 criterion; adaptable)

| Dataset | What it has | Fails | How to adapt |
|---|---|---|---|
| **AppleMOTS** (de Jong et al., RA-L 2022) | UAV + wearable video of apples; ~86,000 masks, 1,700 MOTS frames; Zenodo 5939726 (5.7 GB, open). Baselines TrackR-CNN/PointTrack: MOTSP 80.4, sMOTSA 38.7, MOTSA 52.9;\[25\]\[26\] bbox variant "AppleMOT"\[27\] | Criterion 1 **unverified**: no documented revisits; very likely single passes (inference) | Use as a short-term association / SfM-landmark sanity check. Inspect trajectories for incidental re-entry |
| **Fuji-SfM** (Gené-Mola et al., 2020) | 582 motion-sequence images (291 per row side) of 11 trees; 1,455 apples with 3D GT positions; camera poses + point cloud; 6.43 GB, research/education only\[28\]\[29\] | No per-frame track IDs (2D masks only on 288 separate images). The two row sides form a weak "revisit" | Excellent for validating the **landmark** stage (Gaussian vs. 3D GT apple position). Project 3D GT into frames to synthesize track GT |
| **PFuji-Size** | 6 trees, 615 apples, multi-position photos, SfM/MVS point clouds, per-apple 3D instance + diameter GT\[30\] | Image sets, not tracking video; no 2D IDs | Same as Fuji-SfM |
| **MinneApple** (Häni et al., RA-L 2020) | 1,000 images / 41k instance masks extracted every 5th frame from phone video of one side of a row; counting patches\[1\]\[31\]\[32\]\[33\] | No track IDs; single pass | Not recommended |
| **2023 UAV vineyard set** (Ariza-Sentís, Data in Brief 2023) | 40 UAV videos, 29 MOTS-annotated\[34\] | Each flight records one side of one row (single pass)\[2\] | Pair with GrapeMOTS |
| **BUP20 / BUP-ST20** (Univ. Bonn, CKA glasshouse) | Sweet pepper; PATHoBot at 0.2 m/s; 3× D435i RGB-D + IMU + wheel odometry; 10 sequences over 6 rows on two dates (2020-09-24, 2020-10-01); ~70 GB. BUP20 has instance masks on sampled frames + per-row counts from three annotators. BUP-ST20 (bonndata doi 10.60507/FK2/NUMVO1) adds weak spatial-temporal labels for MOT/VIS\[6\]\[35\]\[36\]\[37\]\[38\] | No dense persistent-ID GT across dates; within a sequence it is a single pass | Treat the two dates as a revisit: label cross-date fruit correspondences (note ripening changes appearance — a useful stress test for map-frame cues). Reprojection-based counting baseline within 4.1% of GT\[39\]\[40\] |
| **BUTom21 / BUTom-ST21** (Halstead et al., arXiv 2607.14934) | Tomato, robot video in Bonn glasshouse, VIS/MOT pseudo-labels\[41\]\[42\] | Pseudo-labels; revisits not documented | Same as BUP-ST20 |
| **SB20** (Bonn) | Sugar beet, BonnBot-I RGB-D; tracking set of 4 rows with 1,151–1,525 images\[6\]\[43\] | Single pass; count-oriented | Low priority |
| **Tomato 3D tracking** (Rapado-Rincón et al., arXiv 2211.02760 / MinkSORT) | ABB arm + L515 RGB-D; 700 images from 100 viewpoints of 7 plants on a semi-cylindrical path; bboxes + tracking IDs (3,137 test instances)\[44\]\[45\] | Continuous sweep, not a revisit after a gap; public availability not confirmed | Strong multi-view 3D-MOT testbed if obtainable |
| **Sugar Beets 2016** (Chebrolu et al., IJRR 2017) | BoniRob over ~3 months; RGB-NIR, Kinect, RTK-GPS, lidar; repeated field passes\[46\]\[47\] | Semantic labels only; "no straightforward object labels for each individual plants" (LFSD authors)\[15\]\[46\] | Multi-session revisits plus RTK make it ideal for building a cross-day plant re-ID benchmark yourself |
| **PhenoBench** (Weyler et al., T-PAMI 2024) | UAV sugar-beet images on 3 dates (2020-05-15, 05-26, 06-05); temporally consistent crop IDs; 583 unique crop plants, 496 on all three dates; RTK-initialized bundle adjustment; association by Hungarian matching with a 15 cm threshold\[48\] | Released as patches/orthoimagery, not tracking video; crops grow non-rigidly | Best public **cross-session plant re-ID** GT; good for testing the map-frame prior in isolation |
| **CitrusFarm** (Teng et al., ISVC 2023) | 7 sequences, 3 citrus fields, 1.7 h, 7.5 km, 1.3 TB; stereo RGB-D, mono/NIR/thermal, LiDAR, IMU, RTK; AWS open bucket\[49\]\[50\]\[51\] | No object annotations\[52\] | Route overlaps give revisits; you would have to label trees/fruit |
| **BLT (Bacchus Long-Term)** | Vineyard multi-session, repeated traversals, RGB-D/LiDAR/RTK\[53\] | No object labels | As above |
| **4D peanut crop monitoring** (Dong et al., arXiv 1610.02482) | 23 sessions over 89 days, 1280×960 at 7.5 Hz, RTK-GPS\[54\] | No object labels | As above |
| **Pheno4D, LAST-Straw** | Spatio-temporal plant point clouds (organ-level)\[55\]\[56\] | Not image-video MOT | Out of scope |

I did not verify KFuji, MAgro, TreeScope, the Rosario soybean dataset (six SLAM sequences, no object labels per secondary mentions), cotton/sorghum/maize video counting, or mango video datasets in this pass. Treat them as unassessed.

## Details — Other Domains

| Dataset | Domain / objects | Capture & 3D | Revisits (how known) | GT | Verdict |
|---|---|---|---|---|---|
| **Aria Digital Twin** (Pan et al., ICCV 2023) | Egocentric indoor (apartment, office); 398 instances, **324 stationary**, 74 dynamic\[57\] | Aria glasses: 2 mono + RGB + IMU; continuous 6DoF device poses, depth maps, full digital-twin scene models\[58\] | Inferred: wearers move around two rooms over 200 sequences, so static objects re-enter view repeatedly. A per-instance static flag (−1 = static) helps filter\[59\] | 6DoF object poses, 2D instance segmentation, 2D/3D boxes — persistent IDs derivable\[57\] | **Strong fit** (all 3 criteria; filter to static instances) |
| **KITTI-360** (Liao et al., T-PAMI 2022) | Suburban driving; static instances (parked cars, poles, signs, buildings) | Stereo + fisheye + 360° laser; GPS/IMU poses; ~300k images\[60\]\[61\] | Liao et al. state that "a static primitive b^n is used at all timestamps (if visible)", giving each static object one instance ID across the entire sequence and across 2D/3D. Repeated routes are documented by a follow-up paper (KD360-VoxelBEV, arXiv 2512.15311), whose static-box filtering "resulted in missing objects on repeated routes" | Dense 2D/3D semantic + instance labels; 3D boxes\[60\] | **Strong fit** for parked-vehicle / road-furniture tracking with long gaps |
| **3RScan** (Wald et al., ICCV 2019) | Indoor rooms re-scanned over time | 1,482 RGB-D scans (Tango) of 478 scenes; camera poses + meshes\[8\]\[62\] | **Documented** rescans with instance associations and rigid transforms between scans\[63\] | 3D instance segmentation + cross-scan correspondences; 2D renderings in RIO10\[62\] | **Partial**: excellent cross-session association, but some objects *move* between scans (criterion 3 holds only for unchanged objects). Hidden test server no longer maintained\[63\]\[64\] |
| **EgoObjects** (Zhu et al., ICCV 2023) | Egocentric household objects | 9K+ videos, 114K annotated frames, no poses released\[65\] | **Documented** "same object instance captured under diverse conditions" across multiple videos\[65\] | Boxes + 14K instance IDs on *sampled* frames\[66\]\[67\] | **Near-miss**: instance re-ID across sessions, but sparse frames; run SfM per video yourself |
| **HOI4D** | Egocentric hand-object interaction | RGB-D, camera poses, scene point clouds\[68\] | Not the point | Object poses, panoptic segmentation\[68\] | **Fails 3** (objects are manipulated) |
| **ScanNet / ScanNet++, TUM RGB-D, Replica** | Indoor RGB-D reconstruction | Poses + meshes | ScanNet includes rescans of some rooms (general knowledge, not verified this pass) | 3D instance labels (ScanNet/ScanNet++), none for TUM | **Plausible** via projected 3D instances; verify rescan-level instance correspondence before use |
| **nuScenes / Waymo / Oxford RobotCar** | Driving | Multi-sensor + poses | RobotCar is multi-traversal but has no object tracks; nuScenes/Waymo scenes are ~20 s clips with few loops (general knowledge) | 3D tracks (nuScenes/Waymo) | **Weak**: revisits rare except across logs |
| **TAO, TREK-150, OVTrack-style benchmarks** | General / egocentric object tracking | Video, no poses | Not structured around revisits | Boxes / IDs | **Fail 3** (mostly moving targets) |
| **Retail (MMPTrack-style), shelf monitoring** | Shoppers, fixed multi-camera\[11\]\[12\] | Fixed cameras | — | Person tracks | **Fail 2 & 3**; no public moving-camera shelf-product MOT found |
| **Warehouse / logistics** | Pallets, boxes | — | — | — | No suitable public dataset found |
| **Camera traps** | Wildlife, fixed camera | Static viewpoint means no baseline for SfM | Animals revisit, the camera does not | Detections, sometimes individual IDs | **Fail 2 & 3**. Only stationary background is reconstructable; a fixed camera makes image-space and map-frame position equivalent, so MapSORT adds nothing |

## Recommendations

**Ranked shortlist (best fit to criteria 1–3 first):**

1. **LettuceMOT** *(agricultural)*: explicit B&F/O&I revisits, 10 FPS overlapping video, MOT GT, existing baselines including your LocalizeSORT numbers.\[14\] It is the cleanest test of long-absence revival.
2. **MOrangeT** *(agricultural)*: documented fruit re-entry, COLMAP reconstructions in the release, MOT16 GT, and a 3D-relocalization ablation.\[4\]\[21\]\[69\] It is the closest published analogue to MapSORT and tests a second crop and domain.
3. **LFSD** *(agricultural)*: closed loops over hundreds of metres with RGB-D + RTK + IMU.\[7\]\[15\] It allows both sensor-based and feed-forward landmark variants. First verify in mot.txt whether IDs persist across the loop; if they do not, relabel the first-ridge revisit.
4. **LettuceMOTS** *(agricultural, reference)*: keep as the primary MOTS benchmark for continuity.
5. **Aria Digital Twin** *(non-agricultural)*: 324 stationary instances with 6DoF GT and device poses.\[9\] It lets you isolate landmark error from association error and shows the method generalizes to indoor AR.
6. **KITTI-360** *(non-agricultural)*: sequence-wide static instance IDs plus repeated routes. It is the best outdoor large-scale "parked object" test.
7. **GrapeMOTS** *(agricultural)*: multi-perspective UAV passes with persistent bunch IDs.\[3\] It tests viewpoint-change revival, where appearance fails hardest.
8. **3RScan** *(non-agricultural)*: cross-session association. Report only on unchanged objects, or use it to test robustness when the stationarity assumption is violated.
9. **BUP20 / BUP-ST20** *(agricultural)*: two-date glasshouse revisits with RGB-D + odometry. It needs cross-date ID labels.
10. **PhenoBench** *(agricultural)*: cross-date plant IDs.\[70\] Use it for the re-ID component in isolation.

**Practical next steps:**
- Build a common harness that evaluates with HOTA/AssA/IDF1 plus count error on items 1–4 and 7. Report revisit-only metrics, such as association accuracy on tracks absent for more than N frames, because aggregate HOTA dilutes the regime MapSORT targets.
- Use Fuji-SfM/PFuji-Size and Aria Digital Twin to quantify landmark accuracy separately from tracking.
- For multi-session revisits, concatenate BUP20 dates or Sugar Beets 2016 sessions and register them via RTK/odometry. Label only a subset of cross-session correspondences.

## Caveats
- Licenses for LettuceMOT, LFSD and MOrangeT data are not explicitly stated beyond CC BY on the articles.\[14\] Confirm before redistribution.
- LFSD ID persistence across loops, AppleMOTS trajectory structure, and ADT revisit frequency are inferences, not documented facts.
- MOrangeT metrics come from the arXiv preprint; the published version may differ slightly.
- Statements about ScanNet rescans, nuScenes/Waymo/RobotCar, TAO/TREK-150 and camera traps come from general knowledge and were not re-verified in this research pass.
- The LocalizeSORT numbers are from your unpublished revision draft, so they are not citable as peer-reviewed results.

## Sources

1. [MinneApple - Dataset Ninja](https://datasetninja.com/minne-apple)
2. [Dataset on UAV RGB videos acquired over a vineyard including bunch labels for object detection and tracking - Research Portal - Wageningen University & Research](https://research.wur.nl/en/publications/dataset-on-uav-rgb-videos-acquired-over-a-vineyard-including-bunc/)
3. [GrapeMOTS: UAV vineyard dataset with MOTS grape bunch annotations recorded from multiple perspectives for enhanced object detection and tracking - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11063988/)
4. [Multiple orange detection and tracking with 3-D fruit relocalization and neural-net based yield regression in commercial sweet orange orchards - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0168169924005908)
5. mcp\_\_Google\_Drive\_\_get\_file\_metadata
6. [Virtual Temporal Samples for Recurrent Neural Networks: applied to semantic segmentation in agriculture](https://arxiv.org/pdf/2106.10118)
7. [LFSD: a VSLAM dataset with plant detection and tracking in lettuce farm](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10497103/)
8. [CrossOver: 3D Scene Cross-Modal Alignment](https://arxiv.org/pdf/2502.15011)
9. [Paper page - Aria Digital Twin: A New Benchmark Dataset for Egocentric 3D Machine Perception](https://huggingface.co/papers/2306.06362)
10. [KITTI-360: A Novel Dataset and Benchmarks for Urban ...](https://www.cvlibs.net/publications/Liao2022PAMI.pdf)
11. [MCTR: Multi Camera Tracking Transformer](https://arxiv.org/pdf/2408.13243)
12. [On-the-Fly Homographies Calibration for Multi-Camera Tracking](https://arxiv.org/pdf/2609.18582)
13. <https://arxiv.org/html/2306.13518>
14. [LettuceMOT: A dataset of lettuce detection and tracking with re-identification of re-occurred plants for agricultural robots - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9716067)
15. [https://www.frontiersin.org/journals/plant-science/articles/ ...](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2023.1175743/xml)
16. [LFSD: a VSLAM dataset with plant detection and tracking in lettuce farm - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10497103/)
17. [LFSD-Dataset](https://ieee-dataport.org/documents/lfsd-dataset)
18. [GitHub - wangshuo9707/LFSD-Dataset](https://github.com/wangshuo9707/LFSD-Dataset)
19. [(PDF) A PIPELINE FOR MULTIPLE ORANGE DETECTION AND TRACKING WITH 3-D FRUIT RELOCALIZATION AND NEURAL-NET BASED YIELD REGRESSION IN COMMERCIAL CITRUS ORCHARDS A PREPRINT](https://www.researchgate.net/publication/376894201_A_PIPELINE_FOR_MULTIPLE_ORANGE_DETECTION_AND_TRACKING_WITH_3-D_FRUIT_RELOCALIZATION_AND_NEURAL-NET_BASED_YIELD_REGRESSION_IN_COMMERCIAL_CITRUS_ORCHARDS_A_PREPRINT)
20. [3D Orange Detection & Yield Prediction](https://www.emergentmind.com/papers/2312.16724)
21. <https://arxiv.org/pdf/2312.16724>
22. [Data in Brief 54 (2024) 110432 Contents lists available at ScienceDirect](https://edepot.wur.nl/657197)
23. [MOTS-annotated UAV Vineyard Dataset captured using Multiple Perspectives to avoid Leaf Occlusion for Object Detection and Tracking](https://zenodo.org/records/10625595)
24. [Occlusion-Aware Multi-Object Tracking in Vineyards via SAM-Based Visibility Modeling](https://doi.org/10.3390/electronics15030621)
25. [Dataset for benchmarking Multiple Object Tracking and Segmentation (MOTS) in an apple orchard field. - Wageningen University & Research](https://research.wur.nl/en/datasets/dataset-for-benchmarking-multiple-object-tracking-and-segmentatio)
26. [APPLE MOTS: Detection, Segmentation and Tracking of Homogeneous Objects Using MOTS](https://ieeexplore.ieee.org/document/9857971/)
27. [Multi-Object Tracking in Agricultural Applications using a Vision Transformer for Spatial Association - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0168169924007701)
28. [Fuji-SfM dataset: A collection of annotated images and point clouds for Fuji apple detection and location using structure-from-motion photogrammetry - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7184157/)
29. [Fuji-SfM dataset](https://www.grap.udl.cat/en/publications/fuji-sfm-dataset/)
30. [PFuji-Size dataset: A collection of images and photogrammetry-derived 3D point clouds with ground truth annotations for Fuji apple detection and size estimation in field conditions](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8633858/)
31. [MinneApple: A Benchmark Dataset for Apple Detection and Segmentation](https://www.researchgate.net/publication/338495204_MinneApple_A_Benchmark_Dataset_for_Apple_Detection_and_Segmentation)
32. [\[1909.06441\] MinneApple: A Benchmark Dataset for Apple Detection and Segmentation](https://ar5iv.labs.arxiv.org/html/1909.06441)
33. [\[1909.06441\] MinneApple: A Benchmark Dataset for Apple Detection and Segmentation](https://arxiv.org/abs/1909.06441)
34. [Dataset on UAV RGB videos acquired over a vineyard including bunch labels for object detection and tracking](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9813505/)
35. [Forecasting the Number of Harvest-ready Fruits of Sweet Peppers Using Multimodal Time-Series Data](https://arxiv.org/pdf/2607.19975)
36. [BUP-ST20: Weakly Labelled Spatial Temporal Sweet Pepper Data - bonndata](https://bonndata.uni-bonn.de/dataset.xhtml?persistentId=doi%3A10.60507%2FFK2%2FNUMVO1&version=&q=&fileTypeGroupFacet=&fileAccess=&fileSortField=date&tagPresort=false)
37. [BUP20 - Dataset Ninja](https://datasetninja.com/bup20)
38. [GitHub - Agricultural-Robotics-Bonn/reproj\_rnn: Explicitly incorporating spatial information to recurrent networks for agriculture · GitHub](https://github.com/Agricultural-Robotics-Bonn/reproj_rnn)
39. [PATHoBot: A Robot for Glasshouse Crop Phenotyping and Intervention](https://arxiv.org/pdf/2010.16272)
40. [PATHoBot: A Robot for Glasshouse Crop Phenotyping and Intervention](https://www.researchgate.net/publication/355432851_PATHoBot_A_Robot_for_Glasshouse_Crop_Phenotyping_and_Intervention)
41. [Still image and spatial-temporal tomato data enabling detection, segmentation, tracking, and video-instance segmentation using strong and weak labels](https://arxiv.org/html/2607.14934)
42. [\[2607.14934\] Still image and spatial-temporal tomato data enabling detection, segmentation, tracking, and video-instance segmentation using strong and weak labels](https://arxiv.org/abs/2607.14934)
43. [https://www.frontiersin.org/journals/plant-science/articles/ ...](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2021.786702/xml)
44. [MinkSORT: A 3D deep feature extractor using sparse convolutions to improve 3D multi-object tracking in greenhouse tomato plants](https://arxiv.org/pdf/2307.05219)
45. [Development and evaluation of automated localisation and reconstruction of all fruits on tomato plants in a greenhouse based on multi-view perception and 3D multi-object tracking](https://arxiv.org/pdf/2211.02760)
46. [Frontiers](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2023.1175743/full)
47. [Agricultural robot dataset for plant classification, localization and mapping on sugar beet fields](https://www.semanticscholar.org/paper/Agricultural-robot-dataset-for-plant-localization-Chebrolu-Lottes/88a0c58dc2d79fc1f191bafebbcb1142791873d9)
48. [Dataset Card for PhenoBench — FiftyOne 1.22.0 documentation](https://docs.voxel51.com/dataset_zoo/datasets_hf/phenobench_raw.html)
49. [CitrusFarm Dataset - Registry of Open Data on AWS](https://registry.opendata.aws/citrus-farm/)
50. [GitHub - UCR-Robotics/Citrus-Farm-Dataset: IROS/ISVC 2023: Multimodal Dataset for Localization, Mapping and Crop Monitoring in Citrus Tree Farms · GitHub](https://github.com/UCR-Robotics/Citrus-Farm-Dataset)
51. [CitrusFarm Dataset](https://ucr-robotics.github.io/Citrus-Farm-Dataset/)
52. [Multimodal Dataset for Localization, Mapping and Crop Monitoring in Citrus Tree Farms](https://www.researchgate.net/publication/376103815_Multimodal_Dataset_for_Localization_Mapping_and_Crop_Monitoring_in_Citrus_Tree_Farms)
53. [GitHub - ricber/digital-agriculture-datasets: Open datasets for training and benchmarking AI and robotic systems in agriculture and off-road environments · GitHub](https://github.com/ricber/digital-agriculture-datasets)
54. [4D Crop Monitoring: Spatio-Temporal Reconstruction for Agriculture](https://arxiv.org/pdf/1610.02482)
55. [Pheno4D: A spatio-temporal dataset of maize and tomato plant point clouds for phenotyping and advanced plant analysis](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0256340)
56. [Lincoln's Annotated Spatio-Temporal Strawberry Dataset (LAST-Straw)](https://arxiv.org/pdf/2403.00566)
57. [Project Aria Grand Challenges: ADT 3D Object Detection ...](https://eval.ai/web/challenges/challenge-page/2093/overview)
58. [\[2306.06362\] Aria Digital Twin: A New Benchmark Dataset for Egocentric 3D Machine Perception](https://arxiv.org/abs/2306.06362)
59. [Data Format](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset/data_format)
60. [KITTI-360: A Novel Dataset and Benchmarks for Urban Scene Understanding in 2D and 3D](https://arxiv.org/html/2109.13410v1)
61. [KITTI Dataset Benchmark](https://www.emergentmind.com/topics/kitti-dataset)
62. [Beyond Controlled Environments: 3D Camera Re-Localization in Changing Indoor Scenes](https://arxiv.org/pdf/2008.02004)
63. [Living Scenes: Multi-object Relocalization and Reconstruction in Changing 3D Environments](https://arxiv.org/html/2312.09138v2)
64. [Objects Can Move: 3D Change Detection by Geometric Transformation Constistency](https://arxiv.org/pdf/2208.09870)
65. [GitHub - facebookresearch/EgoObjects: \[ICCV2023\] EgoObjects: A Large-Scale Egocentric Dataset for Fine-Grained Object Understanding](https://github.com/facebookresearch/EgoObjects)
66. [\[2309.08816\] EgoObjects: A Large-Scale Egocentric Dataset for Fine-Grained Object Understanding](https://arxiv.org/abs/2309.08816)
67. [3rd Continual Learning Workshop Challenge on Egocentric Category and Instance Level Object Understanding](https://arxiv.org/pdf/2212.06833)
68. [SpatialBench: Is Your Spatial Foundation Model an All-Round Player?](https://arxiv.org/pdf/2605.27367)
69. [Towards Practical Precision Agriculture: Real-Time Fruit Detection and Video Analytics on Embedded Edge Hardware](https://arxiv.org/html/2609.13551v1)
70. [PhenoBench: A Large Dataset and Benchmarks for Semantic Image Interpretationin the Agricultural Domain](https://arxiv.org/html/2306.04557v2)
