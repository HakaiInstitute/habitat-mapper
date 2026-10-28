# Performance Metrics

Each [model page](index.md) reports four standard segmentation metrics per class. All are between 0 and 1, and higher
is better.

| Metric        | In plain terms                                                                         |
|---------------|----------------------------------------------------------------------------------------|
| **IoU**       | How much the predicted and hand-labelled areas overlap. The strictest single measure.  |
| **Precision** | Of the pixels the model labelled as this class, the fraction that really were.        |
| **Recall**    | Of the pixels that really were this class, the fraction the model found.              |
| **F1**        | A balance of precision and recall.                                                     |

The metrics are calculated on validation imagery held out from training. They describe performance on imagery
similar to the training data, which may differ from yours.

## Definitions

**Definitions in terms of pixel sets:**

- Let $A$ equal the set of human-labelled pixels.
- Let $B$ be defined as the set of pixel labels predicted by the model.
- Let $A_i$ and $B_i$ be the sets of pixels for a particular class of interest, $i$,
  from labels $A$ and $B$, respectively.

**Definitions in terms of true and false postive/negative classes:**

For class $i$:

- Let $TP_i$ be the true positives.
- Let $FP_i$ be the false positives.
- Let $TN_i$ be the true negatives.
- Let $FN_i$ be the false negatives.

**IoU**

:   The "intersection over union", also called the "Jaccard Index". Defined as:

$$
    IoU_i (A,B) = \frac{|A_i \cap B_i|}{|A_i \cup B_i|} = \frac{TP_i}{TP_i + FP_i + FN_i}
$$

**Precision**

:   The ratio of correct predictions for a class to the count of predictions of that
class:

$$
    Precision_i = \frac{|A_i \cap B_i|}{|A_i|} = \frac{TP_i}{TP_i + FP_i}
$$

**Recall**

:   The ratio of correct predictions for a class to the count of actual instances of
that class:

$$
    Recall_i = \frac{|A_i \cap B_i|}{|B_i|} = \frac{TP_i}{TP_i + FN_i}
$$

**F1**

:   The harmonic mean of precision and recall for a class, providing a single metric that balances both:

$$
    F1_i = 2 \cdot \frac{Precision_i \cdot Recall_i}{Precision_i + Recall_i} = \frac{2TP_i}{2TP_i + FP_i + FN_i}
$$
