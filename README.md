## Attention-dominant global phenotypes

<p align="center">
  <img src="figures/attention_dominant_cluster_spatial_validation.png" width="900">
</p>

This figure illustrates how the model focuses on a small subset of histomorphological phenotypes during inference.

- **Panel A** ranks global clusters by the proportion of high-attention tiles.
- **Panel B** maps the dominant clusters back to the whole-slide image and compares them with the attention heatmap.

Using a high-attention threshold of **0.833**, we found that four global clusters (**3, 16, 10, and 8**) accounted for more than half of all high-attention tiles. Their projected spatial regions showed strong overlap with the main attention hotspots, indicating that the model does not attend to tissue uniformly, but preferentially focuses on a limited set of tumour-centred phenotypes with high structural complexity.
