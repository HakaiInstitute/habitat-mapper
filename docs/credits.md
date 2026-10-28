# Credits

Habitat-Mapper was developed at the [Hakai Institute](https://hakai.org), a marine research organization based in
British Columbia, Canada. The Hakai Institute led development of the software and contributed most of the training
data for the kelp, mussel and gooseneck barnacle models.

We're grateful to everyone who has helped improve Habitat-Mapper with code, data, bug reports, testing and feedback.
Thank you!

## Team

| Name                | Role                                          |
|:--------------------|:----------------------------------------------|
| **Taylor Denouden** | Machine learning and software development     |
| **Luba Reshitnyk**  | Scientific application (kelp), data collation |
| **Will McInnes**    | Data collation and labelling (mussels)        |
| **Alyssa Gehman**   | Scientific application (mussels)              |

## Training data contributors

### The Nature Conservancy of California

[The Nature Conservancy of California](https://www.nature.org/en-us/about-us/where-we-work/united-states/california/)
provided drone imagery used to train the [`kelp-rgb` and `kelp-rgbi`](models/kelp-drone.md) models. This
high-resolution data was instrumental in improving the models' accuracy and robustness.

### Dr. Katherine Cavanaugh

[Dr. Katherine Cavanaugh](https://scholar.google.com/citations?user=Luu6YqYAAAAJ&hl=en) *et al.* contributed model
development experience and expert annotations of Planet Labs satellite imagery, which were used to train the
[`kelp-ps8b`](models/kelp-satellite.md#kelp-ps8b) model. We're grateful for her contributions and support of this
project. See [How to Cite](cite.md#the-models) for the associated paper.

## Model contributors

### Dr. Mohsen Ghanbari

[Dr. Mohsen Ghanbari](https://scholar.google.com/citations?user=9wytH0kAAAAJ&hl=en) *et al.* developed the
[`kelp-skema`](models/kelp-satellite.md#kelp-skema) models as part of a postdoctoral project at
[Spectral Lab](https://uvicspectral.com/). The models were integrated into Habitat-Mapper from the original code at
[github.com/m5ghanba/skema](https://github.com/m5ghanba/skema/). The work is described in
[Ghanbari *et al.* (2026)](https://doi.org/10.3390/rs18203604). See [How to Cite](cite.md#the-models).

## Contact

[Email us](mailto:habitat.mapper.support@hakai.org) or
[open an issue on GitHub](https://github.com/HakaiInstitute/habitat-mapper/issues).
