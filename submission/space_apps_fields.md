# Space Apps project page: text for each field

Final text, ready to paste. Known limit: the summary field allows 1,500 characters; this summary has about 1,130. The limits of the other fields are not known.

---

## Project Name

LunaTerra

## High-Level Project Summary

We developed a deep-learning prototype that uses lunar terrain to estimate annual maximum surface temperatures near the Moon's south pole. Using NASA's LRO Diviner Polar Resource Product, our augmented U-Net achieved a mean absolute error of 7.3 K against model-derived reference temperatures on geographically held-out test areas.

We also applied terrain and temperature screening rules to the NASA reference data, identifying 6,038 km² of candidate zones with gentle slopes, warm surfaces and proximity to potential ice-preserving cold traps. These are preliminary screening results, not confirmed ice deposits or approved landing sites.

To address the Earth-analog challenge, we propose Haughton Crater in Canada for polar crater exploration and University Valley in Antarctica for studying dry ground over shallow ice. Each reproduces selected features, but neither matches the Moon's vacuum or extreme cold-trap temperatures.

By connecting lunar mapping with evidence-based Earth analog selection, our project helps explain which conditions researchers could test on Earth and which require further lunar investigation.

## Link to Project Demo

https://github.com/aorosoeon/lunar-south-pole-base-zones/blob/main/submission/space_apps_deck.pdf

## Link to Final Project

https://github.com/aorosoeon/lunar-south-pole-base-zones

## Detailed Project Description

**What it does.** The project reads lunar south-pole terrain and predicts the annual maximum surface temperature for every 0.5 km cell. From temperatures and terrain it marks:
- **cold traps**, where the annual maximum stays below 110 K, our screening value for long-term surface-ice stability (after Vasavada et al. 1999);
- **candidate base zones**, which are gentle (8° or less), sunlit (annual maximum of 200 K or more) and within 2 km of a cold trap. These cutoffs are our screening choices, not NASA rules.

**How it works.**
1. We parsed the NASA Diviner Polar Resource Product: 2,880,000 triangles on a 1,200 × 1,200 grid covering 600 × 600 km around the pole.
2. We computed terrain features from the triangle geometry: elevation, slope, which way the ground faces, and latitude.
3. We split the map into 50 km blocks for training, validation and test, with 5 km buffers around the validation and test blocks. The model is scored on areas whose temperatures it never saw in training.
4. We compared four models: latitude only, an MLP that looks at one triangle at a time, a U-Net CNN, and the same U-Net with rotation and flip augmentation. The augmentation is physically valid because the Sun circles the pole over a year. It cut the test error from 12.9 K to 7.3 K and greatly reduced overfitting to the training craters.
5. With NASA's reference temperatures, candidate base zones cover 6,038 km² in 1,640 patches. We applied the same rule to the CNN's predictions, with two thresholds tuned on validation blocks. On the test blocks, the CNN finds 74% of the cold-trap area and 53% of the base-zone area from terrain alone.

**Benefits.** Once trained, the model gives a thermal screening map in seconds from elevation data alone. It narrows the search for base sites, but it cannot choose one. The same list of traits tells analog researchers on Earth what to match.

**Earth analogs.** We propose Haughton Crater in Canada's High Arctic as a partial analog for exploring polar crater terrain and ice-bearing ground. University Valley in Antarctica provides a complementary setting for studying dry soil over shallow ground ice. These sites can support field tests of exploration and ice-access methods, but neither reproduces the lunar combination of permanent shadow, vacuum and surface temperatures below 110 K. Even the coldest satellite-observed Antarctic snow surfaces reported by Scambos et al. (2018), about 175 K, are far warmer than our <110 K lunar cold-trap threshold. They are literature-selected analog candidates, not sites validated or ranked by our lunar neural network.

**What we hope to achieve.** Next, we want to:
- test whether a graph neural network on the original triangle mesh sharpens the shadow edges;
- add a horizon-height input for distant shadow casters;
- test the model on the north pole;
- build an Earth analog comparison from illumination, climate and ground-ice data.

**Tools.** Python 3.12, PyTorch (U-Net trained on an Apple M5 Pro GPU, checked against the CPU), NumPy, pandas, SciPy, scikit-learn and Matplotlib. Slides are in HTML, exported to PDF.

## Use of Super Intelligence (SI)

The team chose the project idea, dataset, research questions and analysis decisions. Claude (Anthropic) wrote most of the Python code, created the slide layout and drafted text under Arsen's direction. Feyi used ChatGPT/Codex (OpenAI) to find and inspect scientific sources, compare Earth analog sites, check claims and consistency across the repository and slides, and draft contribution and disclosure text. The maps and plots were produced from the NASA data by code, rather than generated as synthetic images. AI assistance does not constitute independent validation of the model or its conclusions.

## NASA Data

- **LRO Diviner Lunar Radiometer Experiment, Polar Resource Products V2.0** (LRO-L-DLRE-5-PRP-V2.0), south pole table `dlre_prp_south.tab`, from the NASA PDS Geosciences Node: https://pds-geosciences.wustl.edu/lro/urn-nasa-pds-lro_diviner_derived1/data_derived_prp/
  - This is our only data source: terrain geometry, annual maximum temperature (the model target), and ice depth (used only to check the cold-trap rule, never as a model input).
- Fong, T., et al. (2010). *Robotic Follow-Up for Human Exploration.* NASA Technical Reports Server, 20110008151 (source for Haughton Crater). https://ntrs.nasa.gov/citations/20110008151
- Gläser, P., et al. (2017). *Illumination Conditions at the Lunar Poles: Implications for Future Exploration.* NASA Technical Reports Server, 20170007365 (source for the low sun angle). https://ntrs.nasa.gov/citations/20170007365

## Space Agency Partner Data and Other References

- No space agency partner data was used.
- Paige, D. A., et al. (2018). LRO DLRE Level 5 Polar Resource Products V2.0. NASA Planetary Data System.
- Paige, D. A., et al. (2010). Diviner Lunar Radiometer observations of cold traps in the Moon's south polar region. *Science*, 330(6003), 479–482. https://doi.org/10.1126/science.1187726
- Vasavada, A. R., Paige, D. A., & Wood, S. E. (1999). Near-surface temperatures on Mercury and the Moon and the stability of polar ice deposits. *Icarus*, 141(2), 179–193. https://doi.org/10.1006/icar.1999.6175
- Wueller, L., et al. (2026). Assessing potential landing sites with favorable illumination and accessible, potentially volatile-rich permanently shadowed regions within Artemis candidate landing regions. *Journal of Geophysical Research: Planets*, 131(3), e2025JE009434. https://doi.org/10.1029/2025JE009434
- Marinova, M. M., et al. (2022). Climate and energy balance of the ground in University Valley, Antarctica. *Antarctic Science*, 34(2), 144–171. https://doi.org/10.1017/S0954102022000025
- Scambos, T. A., et al. (2018). Ultralow surface temperatures in East Antarctica from satellite thermal infrared mapping: the coldest places on Earth. *Geophysical Research Letters*, 45(12), 6124–6133. https://doi.org/10.1029/2018GL078133
- Software: PyTorch, NumPy, pandas, SciPy, scikit-learn, Matplotlib (all open source).
