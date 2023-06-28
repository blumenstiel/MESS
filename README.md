# MESS – Multi-domain Evaluation of Semantic Segmentation

[[Website](https://blumenstiel.github.io/mess-benchmark/)] [[arXiv](https://arxiv.org/abs/2306.15521)]

This is the official toolkit for the MESS benchmark from our Paper "What a MESS: Multi-domain Evaluation of Zero-shot Semantic Segmentation".

The MESS benchmark enables a holistic evaluation of semantic segmentation models on a variety of domains and datasets. The MESS benchmark includes 22 datasets for different domains like medicine, engineering, earth monitoring, biology, and agriculture. We designed this toolkit to be easy to use for new model architectures. We invite others to propose new ideas and datasets for future versions.

We provide a [leaderboard](https://blumenstiel.github.io/mess-benchmark/leaderboard/) with all evaluated models and links to the implementation.

## Usage

To test a new model architecture, download this repository, copy the `mess` directory to your project, and follow the steps in [mess/DATASETS.md](mess/DATASETS.md) for downloading and preparing the datasets.
You can register the datasets to detectron2 by adding `import mess.datasets` to your evaluation code. See [mess/README.md](mess/README.md) for more details.

## TODOs

- [ ] Add preprocessing code for training sets to enable few-shot and supervised settings

## License

This code is released under the [MIT License](LICENSE). The evaluated datasets are released under their respective licenses, see [DATASETS.md](mess/DATASETS.md) for details. Most datasets are limited to non-commercial use only and require a citation which are provided in [mess/datasets.bib](mess/datasets.bib).

## Acknowledgement

We would like to acknowledge the work of the dataset providers, especially for the careful collection and annotation of the datasets. Thank you for making the dataset publicly available!
See [DATASETS.md](mess/DATASETS.md) for more details and links to the datasets. We like to further thank the authors of the evaluated models for their work and providing the model weights.

## Citation

Please cite our paper if you use the MESS benchmark and send us your results to be included in the leaderboard.

```
@article{MESSBenchmark2023,
  title={{What a MESS: Multi-domain Evaluation of Zero-shot Semantic Segmentation}},
  author={Blumenstiel, Benedikt and Jakubik, Johannes and Kühne, Hilde and Vössing, Michael},
  journal={arXiv preprint arXiv:2306.15521},
  year={2023}
}
```
