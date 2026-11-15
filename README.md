# [Project name]

**Mapping candidate base zones at the lunar south pole from terrain, with deep learning**

[Team name] · [Team members] · NASA Space Apps Challenge 2026: *Identify Earth Locations that Analog the Permanent Moon Base Locations and Mars*

A permanent base at the lunar south pole needs three things close together: sunlight, gentle ground, and water ice. Ice survives only in cold traps, the craters that never see the Sun. We trained neural networks to predict the annual maximum surface temperature from terrain alone, using NASA's LRO Diviner polar data. Then we used those temperatures to map candidate base zones: gentle, sunlit ground within 2 km of a cold trap. These zones also define what an Earth analog site has to match.

Demo slides: [`submission/space_apps_deck.pdf`](submission/space_apps_deck.pdf). To present, open `submission/space_apps_deck.html` and use the arrow keys.

## Results

Scores are on test blocks: 50 km areas whose temperatures the models never saw during training.

| Model | Mean absolute error, all | Mean absolute error, cold traps (< 110 K) |
|---|---|---|
| Latitude only (linear) | 30.8 K | 144 K |
| MLP, one triangle at a time | 13.5 K | 82 K |
| U-Net CNN on the grid | 12.9 K | 83 K |
| **U-Net CNN + rotation/flip augmentation** | **7.3 K** | **45 K** |

- The first CNN memorized the training craters. Its cold-trap error was 32 K in the training area and 83 K on test.
- Rotating and flipping the training patches fixed most of this. The rotations also turn the "direction to the pole" input, so every patch stays physically valid. This is allowed because the Sun circles the pole over a year.
- Candidate base zones cover **6,038 km²** in 1,640 patches, which is 1.7% of the map.
- From terrain alone, the CNN finds 74% of the cold-trap area and 53% of the base-zone area in the test blocks. This is good enough to screen terrain, but not to choose a site.

## Data

- **NASA LRO Diviner Polar Resource Product, V2.0**, south pole table `dlre_prp_south.tab` (604.8 MB). Download it from the [PDS Geosciences Node](https://pds-geosciences.wustl.edu/lro/urn-nasa-pds-lro_diviner_derived1/data_derived_prp/) and put it in `data/`.
- The table has 2,880,000 triangles on a 1,200 × 1,200 grid of about 0.5 km cells, covering 600 × 600 km around the pole.
- The temperatures are outputs of a thermal model fitted to Diviner's first mapping year. They are not direct measurements.
- **Data note:** the `ice_depth` column uses a code, -1, that the label file does not document. It appears on 100,319 triangles, all in the coldest areas. We read -1 as "ice is stable at the surface". With that reading, the 110 K cold-trap rule matches the dataset's own ice model with an overlap (IoU) of 0.97. `ice_depth` is never used as a model input.

## How to run

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
cd src
../.venv/bin/python s01_parse.py           # parse, check columns and mesh layout
../.venv/bin/python s02_features_split.py  # features, map orientation check, geographic split
../.venv/bin/python s03_mlp.py             # latitude-only and MLP baselines
../.venv/bin/python s03_cnn.py             # first U-Net (overfits)
../.venv/bin/python s03b_cnn_aug.py        # U-Net with augmentation (about 7 min on an Apple M5 Pro GPU)
../.venv/bin/python s03c_compare_maps.py   # comparison figure
../.venv/bin/python s04_base_zones.py      # base zones, model vs reference
../.venv/bin/python s05_slide_figs.py      # slide figures
```

| Folder | Contents |
|---|---|
| `src/` | One script per step; `common.py` holds the shared split and scoring |
| `figures/` | All maps and plots |
| `data/` | Raw table (not in the repo), derived files, logs, scores, model weights |
| `submission/` | Demo slides (HTML and PDF) and the Space Apps page text |

## Method details

- **Features** come from the triangle geometry: elevation, slope, the north and east parts of the surface normal, and latitude.
- **Split:** the grid is cut into 144 blocks of 50 km, assigned to train, validation or test. Blocks are grouped by cold-trap share, so each split gets its share of cold craters. Training cells within 5 km of a validation or test block are removed. Models may read terrain anywhere, but they learn from training labels only.
- **Base-zone rule:** slope of 8° or less, sunlit (annual max of 200 K or more), and within 2 km of a cold trap (annual max below 110 K), in patches of at least 1 km². The model's two temperature thresholds were tuned on validation blocks only (cold below 150 K, sunlit at 200 K or more), then scored on test blocks.

## Limits

- The targets come from a thermal model, not from measurements, so this is a surrogate model.
- Annual maximum temperature shows peak sunlight, not how many hours of sunlight a place gets.
- The CNN blurs shadow edges, which is exactly where base zones sit.
- Only the south pole is covered so far.

## Use of AI

The team chose the project idea, the data, the analysis questions and the conclusions. Claude (Anthropic), an AI assistant, wrote most of the code in `src/`, the slide layout and drafts of this text, under the team's direction and review. No images are AI-generated: every map and plot is drawn from the data by the code in `src/`.

## References

- Paige, D. A., et al. (2018). *LRO DLRE Level 5 Polar Resource Products V2.0*, LRO-L-DLRE-5-PRP-V2.0. NASA Planetary Data System.
- Paige, D. A., et al. (2010). Diviner Lunar Radiometer observations of cold traps in the Moon's south polar region. *Science* 330. [to verify: page numbers]
- [To verify: source for the 8° landing-slope limit]
- [To verify: source for the 110 K limit for stable surface ice]
