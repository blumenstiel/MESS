# [MESS – Multi-domain Evaluation of Semantic Segmentation](https://blumenstiel.github.io/mess-benchmark/)

This is the official toolkit for the MESS benchmark from the NeurIPS 2023 paper "What a MESS: Multi-domain Evaluation of Zero-shot Semantic Segmentation".
Please visit our [website](https://blumenstiel.github.io/mess-benchmark/) or [paper](https://arxiv.org/abs/2306.15521) for more details.

The MESS benchmark enables a holistic evaluation of semantic segmentation models on a variety of domains and datasets. 
The MESS benchmark includes 22 datasets for different domains like medicine, engineering, earth monitoring, biology, and agriculture. 
We designed this toolkit to be easy to use for new model architectures. We invite others to propose new ideas and datasets for future versions.

The website includes a [leaderboard](https://blumenstiel.github.io/mess-benchmark/leaderboard/) with all evaluated models and links to their implementations.

## Usage

To test a new model architecture, install the benchmark with `pip install mess-benchmark`, and follow the steps in [DATASETS.md](DATASETS.md) for downloading and preparing the datasets.
You can register all datasets by running `import mess.datasets`. See [GettingStarted.md](GettingStarted.md) for more details.

### Zero-shot semantic segmentation

The current version of the MESS benchmark focuses on zero-shot semantic segmentation, and the toolkit is ready to use for this setting.

### Few-shot and many-shot semantic segmentation

Few-shot and many-shot semantic segmentation is not yet supported by the toolkit, but can easily be added based on the provided preprocessing scripts.
Most datasets provide a train/val split that can be used for few-shot or supervised training. 
CHASE DB1 and CryoNuSeg do not provide train data themselves, but use other similar datasets for training (DRIVE and STARE for CHASE DB1 and MoNuSeg for CryoNuSeg).
BDD100K, Dark Zurich, iSAID, and UAVid are evaluated using their official validation split. 
Hence, supervised training may require the train set to be split into a train and val dev split.  

The DRAM dataset only provides an unlabelled train set and would require a style transfer to Pascal VOC for labelled training data.
The WorldFloods train set requires approximately 300Gb of disk space, which may not be feasible for some users.
Therefore, we propose to exclude DRAM and WorldFloods from the few-shot and many-shot settings to simplify the evaluation, called **MESS-20**.

## License

This code is released under the [MIT License](LICENSE). The evaluated datasets are released under their respective licenses, see [DATASETS.md](DATASETS.md) for details. Most datasets are limited to non-commercial use only and require a citation which are provided in [datasets.bib](datasets.bib).

## Acknowledgement

We would like to acknowledge the work of the dataset providers, especially for the careful collection and annotation of the datasets. Thank you for making the dataset publicly available!
See [DATASETS.md](DATASETS.md) for more details and links to the datasets. We like to further thank the authors of the evaluated models for their work and providing the model weights.

## Citation

Please cite our [paper](https://arxiv.org/abs/2306.15521) if you use the MESS benchmark and send us your results to be included in the [leaderboard](https://blumenstiel.github.io/mess-benchmark/leaderboard/).

```
@article{MESSBenchmark2023,
  title={{What a MESS: Multi-Domain Evaluation of Zero-shot Semantic Segmentation}},
  author={Blumenstiel, Benedikt and Jakubik, Johannes and Kühne, Hilde and Vössing, Michael},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
```
