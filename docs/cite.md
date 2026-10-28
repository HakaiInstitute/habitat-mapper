# How to Cite

If you use Habitat-Mapper in research, reports or publications, please cite **the software** and, where listed,
**the paper behind each model you used**. Citations help us show the value of this work and keep it funded.

## The software

=== "APA"

    Denouden, T., & Reshitnyk, L. *Habitat-Mapper* [Computer software]. https://doi.org/10.5281/zenodo.17203205

=== "BibTeX"

    ```bibtex
    @software{Denouden_Habitat-Mapper,
      author = {Denouden, Taylor and Reshitnyk, Luba},
      doi = {10.5281/zenodo.17203205},
      title = {{Habitat-Mapper}},
      url = {https://github.com/HakaiInstitute/habitat-mapper}
    }
    ```

To cite the exact version you used, check `hab --version` and find that release on
[Zenodo](https://doi.org/10.5281/zenodo.17203205). The citation is also in
[`CITATION.cff`](https://github.com/HakaiInstitute/habitat-mapper/blob/main/CITATION.cff), which GitHub shows as
*Cite this repository*.

## The models

Please also state which model and revision you used, for example *`kelp-rgb` revision `20240722`*, and cite or
acknowledge the contributors below.

| Model                                  | Cite or acknowledge                                                                                                             |
|----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| `kelp-rgb`, `kelp-rgbi`                | Acknowledge The Nature Conservancy of California for training imagery.                                                          |
| `kelp-ps8b`                            | Cavanaugh, K.C. *et al.* High-resolution planet Dove data identify local drivers of kelp canopy persistence. *Commun Earth Environ* (2026). [doi:10.1038/s43247-025-03134-y](https://doi.org/10.1038/s43247-025-03134-y) |
| `kelp-skema`                           | Ghanbari, M. *et al.* Deep Learning-Based Mapping of Floating Kelp Forests from Sentinel-2 Imagery. *Remote Sensing* 18(20), 3604 (2026). [doi:10.3390/rs18203604](https://doi.org/10.3390/rs18203604) |
| `mussel-rgb`, `mussel-gooseneck-rgb`   | Software citation above.                                                                                                        |
| `eelgrass-rgb`                         | Software citation above.                                                                                                        |

If you used one of the BC-only `kelp-skema` variants, please also cite the substrate and bathymetry datasets listed
under [Auxiliary data sources](guides/skema/variants.md#variants-at-a-glance).

??? example "BibTeX for `kelp-skema`"

    ```bibtex
    @article{Ghanbari_2026_SKeMa,
      author = {Ghanbari, Mohsen and Ernst, Neil and Denouden, Taylor A. and Reshitnyk, Luba Y. and
                Steffen, Piper and Wachmann, Alena and Mora-Soto, Alejandra and Loos, Eduardo and
                Hessing-Lewis, Margot and Dedeluk, Nic and Costa, Maycira},
      title = {Deep Learning-Based Mapping of Floating Kelp Forests from Sentinel-2 Imagery},
      journal = {Remote Sensing},
      volume = {18},
      number = {20},
      pages = {3604},
      year = {2026},
      doi = {10.3390/rs18203604}
    }
    ```

See [Credits](credits.md) for everyone who contributed.
