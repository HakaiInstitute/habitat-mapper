# Attribution

We are grateful to all contributors who have helped improve Habitat-Mapper with code and data contributions, bug
reports, testing, and feedback. Thank you!

## Development and Support

**Habitat-Mapper** was developed by the [Hakai Institute](https://hakai.org), a marine research organization based in
British Columbia, Canada. The Hakai Institute led the development of this software package and contributed the majority
of machine learning training data used to build the kelp detection and mussel and gooseneck barnacle detection models.

*Have questions or feedback?*

[Send us an email](mailto:habitat.mapper.support@hakai.org) or [file a GitHub issue](https://github.com/HakaiInstitute/habitat-mapper/issues).

## Training Data Contributors

### The Nature Conservancy of California

[The Nature Conservancy of California](https://www.nature.org/en-us/about-us/where-we-work/united-states/california/)
generously provided drone-based kelp imagery used to help train the **kelp-rgb** and **kelp-rgbi** models. This
high-resolution aerial data was instrumental in improving model accuracy and robustness.

### Dr. Katherine Cavanaugh

[Dr. Katherine Cavanaugh](https://scholar.google.com/citations?user=Luu6YqYAAAAJ&hl=en) *et al.* contributed valuable
model development experience as well as expert annotations of Planet Labs satellite imagery that were used to help
train the **kelp-ps8b** model[^1]. We are incredibly grateful for her contributions and support of this project.

## Citation

If you use Habitat-Mapper in your research, please cite the software:

=== "APA"
    Denouden, T., & Reshitnyk, L. Habitat-Mapper [Computer software]. https://doi.org/10.5281/zenodo.17203205

=== "Bibtex"
    ```bibtex
    @software{Denouden_Habitat-Mapper,
    author = {Denouden, Taylor and Reshitnyk, Luba},
    doi = {10.5281/zenodo.17203205},
    title = {{Habitat-Mapper}},
    url = {https://github.com/HakaiInstitute/habitat-mapper}
    }
    ```

[^1]: Cavanaugh, K.C., Cavanaugh, K.C., Berberian, L.A. et al. High-resolution planet Dove data identify local drivers
    of kelp canopy persistence. *Commun Earth Environ* (2026).
    [https://doi.org/10.1038/s43247-025-03134-y](https://doi.org/10.1038/s43247-025-03134-y)
