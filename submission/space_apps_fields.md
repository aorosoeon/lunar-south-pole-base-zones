# Space Apps project page: draft text for each field

Edit before you paste. Items in [brackets] are for you to fill in. Feyi owns the Earth-analog text and the facts marked "verify".

---

## Project Name

[Project name]

## High-Level Project Summary

We built a deep learning model that predicts the thermal environment at the lunar south pole from terrain alone. We then used it to map candidate sites for a permanent base. A base needs sunlight, gentle ground and water ice close together, and ice survives only in cold traps that never see the Sun.

We trained a U-Net neural network on NASA's LRO Diviner polar temperature data (2.88 million terrain elements). On terrain it had never seen, it predicts the annual maximum surface temperature to within 7.3 K on average.

From these temperatures we mapped 6,038 km² of candidate base zones: gentle, sunlit ground within 2 km of a cold trap. These lunar numbers become a checklist for Earth analogs: low sun, gentle ground next to permanent shadow, extreme temperature contrast, and ice held in place by cold. [Feyi: one sentence on the Earth sites you picked.]

## Link to Project Demo

[Public Google Drive link to `space_apps_deck.pdf`, 7 slides. Test it in a private browser window.]

## Link to Final Project

[Public GitHub repository link. Test it in a private browser window.]

## Detailed Project Description

**What it does.** The project reads lunar south-pole terrain and predicts the annual maximum surface temperature for every 0.5 km cell. From the predicted temperatures and the terrain it marks:
- **cold traps**, where the annual maximum stays below 110 K, so water ice can survive;
- **candidate base zones**, which are gentle (8° or less), sunlit (200 K or more) and within 2 km of a cold trap.

**How it works.**
1. We parsed the NASA Diviner Polar Resource Product: 2,880,000 triangles on a 1,200 × 1,200 grid covering 600 × 600 km around the pole.
2. We computed terrain features from the triangle geometry: elevation, slope, which way the ground faces, and latitude.
3. We split the map into 50 km blocks for training, validation and test, with 5 km buffers between them. This keeps the test honest: the model is scored on terrain it has never seen.
4. We compared four models: latitude only, an MLP that looks at one triangle at a time, a U-Net CNN, and the same U-Net with rotation and flip augmentation. The augmentation is physically valid because the Sun circles the pole over a year. It cut the test error from 12.9 K to 7.3 K and stopped the model from memorizing the training craters.
5. We applied the base-zone rule to the reference temperatures and to the model's predictions, and compared the two. From terrain alone, the model finds 74% of the cold-trap area and 53% of the base-zone area in the test blocks.

**Benefits.** Once trained, the model gives a thermal screening map in seconds from elevation data alone. The same checklist of traits can rank terrain on the Moon, and it tells analog hunters on Earth exactly what to match.

**What we hope to achieve.** Next, we want to:
- sharpen the shadow edges with a graph neural network on the original triangle mesh;
- add a horizon-height input for distant shadow casters;
- test the model on the north pole;
- run the same model on Earth terrain to rank analog sites.

**Earth analogs.** [Feyi: 2–4 sentences. Which Earth sites match which trait, and where each analogy breaks down. Air, gravity and radiation always differ, and no Earth surface gets as cold as a lunar cold trap (verify and cite).]

**Tools.** Python 3.12, PyTorch (U-Net trained on an Apple M5 Pro GPU, checked against the CPU), NumPy, pandas, SciPy, scikit-learn and Matplotlib. Slides are in HTML, exported to PDF.

## Use of Super Intelligence (SI)

[Write this in your own words. Factual draft:] The team came up with the project idea, chose the data, and made the analysis decisions and conclusions. We used Claude (Anthropic), an AI assistant, to write most of the Python code, lay out the slides and draft text, under our direction and review. No images are AI-generated: every map and plot is drawn from NASA data by our code. The README states the same.

## NASA Data

- **LRO Diviner Lunar Radiometer Experiment, Polar Resource Products V2.0** (LRO-L-DLRE-5-PRP-V2.0), south pole table `dlre_prp_south.tab`, from the NASA PDS Geosciences Node: https://pds-geosciences.wustl.edu/lro/urn-nasa-pds-lro_diviner_derived1/data_derived_prp/
  - This is our only data source: terrain geometry, annual maximum temperature (the model target), and ice depth (used only to check the cold-trap rule, never as a model input).

## Space Agency Partner Data and Other References

- No space agency partner data was used.
- Paige, D. A., et al. (2018). LRO DLRE Level 5 Polar Resource Products V2.0. NASA Planetary Data System.
- Paige, D. A., et al. (2010). Diviner Lunar Radiometer observations of cold traps in the Moon's south polar region. *Science* 330. [verify page numbers]
- [verify: source for the 8° landing-slope limit]
- [verify: source for the 110 K limit for stable surface ice]
- Software: PyTorch, NumPy, pandas, SciPy, scikit-learn, Matplotlib (all open source).
