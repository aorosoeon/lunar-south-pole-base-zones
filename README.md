# LunaTerra

**Mapping candidate base zones at the lunar south pole with deep learning, and what an Earth analog must match**

Olive Nebulae · Arsen Chuzhykov, Oluwafeyisayo Afolabi · NASA Space Apps Challenge 2026: *Identify Earth Locations that Analog the Permanent Moon Base Locations and Mars*

A permanent base at the lunar south pole needs three things close together: sunlight, gentle ground, and water ice. Surface ice lasts only in cold traps, the shadowed ground that never sees the Sun. We trained neural networks to predict the annual maximum surface temperature from terrain alone, using NASA's LRO Diviner polar data. Then we used temperatures and terrain to map candidate base zones: gentle, sunlit ground within 2 km of a cold trap. These zones also define what an Earth analog site has to match.

Demo slides: [`submission/space_apps_deck.pdf`](submission/space_apps_deck.pdf). To present, open `submission/space_apps_deck.html` and use the arrow keys.

## Results

Scores are on test blocks: 50 km areas whose temperatures the models never saw during training.

| Model | Mean absolute error, all | Mean absolute error, cold traps (< 110 K) |
|---|---|---|
| Latitude only (linear) | 30.8 K | 144 K |
| MLP, one triangle at a time | 13.5 K | 82 K |
| U-Net CNN on the grid | 12.9 K | 83 K |
| **U-Net CNN + rotation/flip augmentation** | **7.3 K** | **45 K** |
| Same, averaged over 8 rotated and flipped copies at prediction time | 6.5 K | 45 K |

- The first CNN overfit the training craters. Its cold-trap error was 32 K in the training area and 83 K on test.
- Rotating and flipping the training patches cut the test cold-trap error to 45 K, while the training-area error stayed at 33 K. The gap shrank from 51 K to 12 K, but it did not close. The rotations also turn the "direction to the pole" input, so every patch stays physically valid. This is allowed because the Sun circles the pole over a year.
- With NASA's reference temperatures, candidate base zones cover **6,038 km²** in 1,640 patches, which is 1.7% of the map.
- From terrain alone, the CNN finds 74% of the cold-trap area and 53% of the base-zone area in the test blocks (recall). For these maps the CNN uses the 8-copy averaged predictions and two thresholds tuned on validation blocks (see Method details). With the reference thresholds, it finds only 18% of the base-zone area.
- This is a first screening step. It narrows the search, but it cannot choose a landing or base site.

## Earth analogs

No Earth site matches the full lunar environment. These two sites match some traits:

- **Haughton Crater, Devon Island, Canada (75°22′N, 89°41′W).** A 20 km polar impact crater with ice-rich permafrost. It matches the low-sun and ground-ice traits in part. It does not have the lunar sun angle, and its ice is buried, not exposed in shadow (Fong et al. 2010).
- **University Valley, McMurdo Dry Valleys, Antarctica (about 77°52′S, 160°43′E).** Dry soil over ground ice at 42 cm depth, with a mean air temperature of −23.4 °C. It matches the ground-ice trait in part. Air moisture and soil cover control the ice, not shadow alone (Marinova et al. 2022).
- **Where every analog breaks down:** air, gravity and radiation differ. The coldest snow surface found by satellite in East Antarctica was −98.6 °C, about 175 K (Scambos et al. 2018). That is far warmer than the lunar cold traps below 110 K.

We chose these sites from the literature. Our model did not rank them.

## Data

- **NASA LRO Diviner Polar Resource Product, V2.0**, south pole table `dlre_prp_south.tab` (604.8 MB). Download it from the [PDS Geosciences Node](https://pds-geosciences.wustl.edu/lro/urn-nasa-pds-lro_diviner_derived1/data_derived_prp/) and put it in `data/`.
- The table has 2,880,000 triangles on a 1,200 × 1,200 grid of about 0.5 km cells, covering 600 × 600 km around the pole.
- The temperatures are outputs of a thermal model fitted to Diviner's first mapping year. They are not direct measurements.
- **Data note:** the `ice_depth` column uses a code, -1, that the label file does not document. It appears on 100,319 triangles, all in the coldest areas. We read -1 as "ice is stable at the surface". This reading is ours and provisional. With it, the 110 K cold-trap rule matches the dataset's own ice model with an overlap (IoU) of 0.97. This shows agreement inside the dataset, not independent confirmation. `ice_depth` is never used as a model input.

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
- **Base-zone rule (our screening choices):** slope of 8° or less, sunlit (annual max of 200 K or more), and within 2 km of a cold trap (annual max below 110 K), in patches of at least 1 km².
  - 8° is our terrain-screening cutoff. Wueller et al. (2026) quote a 0–8° lander orientation limit. A 0.5 km grid cannot certify that a landing spot is safe.
  - 110 K is our screening value for long-term surface-ice stability, after Vasavada et al. (1999).
  - 200 K is our proxy for sunlit ground. It does not measure hours of sunlight.
- **Model thresholds:** the CNN's predictions run warm in cold traps. So its two thresholds were tuned on validation blocks only (cold below 150 K, sunlit at 200 K or more), then scored on test blocks. The 150 K value corrects the model's bias. It is not a new ice-stability limit.

## Limits

- The targets come from a thermal model, not from measurements, so this is a surrogate model.
- Annual maximum temperature shows peak sunlight, not how many hours of sunlight a place gets.
- The CNN blurs shadow edges, which is exactly where base zones sit.
- Only the south pole is covered so far.
- The Earth sites come from the literature. The model does not apply to Earth terrain without new work.

## Use of AI

The team chose the project idea, dataset, research questions and analysis decisions. Claude (Anthropic) wrote most of the Python code, created the slide layout and drafted text under Arsen's direction. Feyi used ChatGPT/Codex (OpenAI) to find and inspect scientific sources, compare Earth analog sites, check claims and consistency across the repository and slides, and draft contribution and disclosure text. The maps and plots were produced from the NASA data by code, rather than generated as synthetic images. AI assistance does not constitute independent validation of the model or its conclusions.

## References

- Paige, D. A., et al. (2018). *LRO DLRE Level 5 Polar Resource Products V2.0*, LRO-L-DLRE-5-PRP-V2.0. NASA Planetary Data System.
- Paige, D. A., et al. (2010). Diviner Lunar Radiometer observations of cold traps in the Moon's south polar region. *Science*, 330(6003), 479–482. https://doi.org/10.1126/science.1187726
- Vasavada, A. R., Paige, D. A., & Wood, S. E. (1999). Near-surface temperatures on Mercury and the Moon and the stability of polar ice deposits. *Icarus*, 141(2), 179–193. https://doi.org/10.1006/icar.1999.6175
- Wueller, L., Berger, L. M., Christopher, H., Sugimoto, K., et al. (2026). Assessing potential landing sites with favorable illumination and accessible, potentially volatile-rich permanently shadowed regions within Artemis candidate landing regions. *Journal of Geophysical Research: Planets*, 131(3), e2025JE009434. https://doi.org/10.1029/2025JE009434
- Gläser, P., Oberst, J., Neumann, G. A., Mazarico, E., et al. (2017). *Illumination Conditions at the Lunar Poles: Implications for Future Exploration.* NASA Technical Reports Server, 20170007365. https://ntrs.nasa.gov/citations/20170007365
- Fong, T., Bualat, M., Deans, M. C., et al. (2010). *Robotic Follow-Up for Human Exploration.* NASA Technical Reports Server, 20110008151. https://ntrs.nasa.gov/citations/20110008151
- Marinova, M. M., et al. (2022). Climate and energy balance of the ground in University Valley, Antarctica. *Antarctic Science*, 34(2), 144–171. https://doi.org/10.1017/S0954102022000025
- Scambos, T. A., et al. (2018). Ultralow surface temperatures in East Antarctica from satellite thermal infrared mapping: the coldest places on Earth. *Geophysical Research Letters*, 45(12), 6124–6133. https://doi.org/10.1029/2018GL078133
