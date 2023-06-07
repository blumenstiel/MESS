# Multi-domain Evaluation of Semantic Segmentation (MESS)

[[Website](https://github.io)] [[arXiv](https://arxiv.org)]

This is the official toolkit for the MESS benchmark from our Paper "What a MESS: Multi-domain Evaluation of Zero-shot Semantic Segmentation".

The MESS benchmark enables a holistic evaluation of semantic segmentation models on a variety of domains and datasets. The MESS benchmark includes 22 datasets for different domains like medicine, engineering, earth monitoring, biology, and agriculture. We designed this toolkit to be easy to use for new model architectures. We invite others to propose new datasets for future versions.

## Usage

To test a new model architecture, download this repository, copy the `mess` directory to your project, and follow the steps in [mess/DATASETS.md](mess/DATASETS.md) for downloading and preparing the datasets.
You can register the datasets to Detectron2 by adding `import mess.datasets` to your evaluation code. See [mess/README.md](mess/README.md) for more details.

## Evaluation

Our evaluation focus on zero-shot transfer models as they are designed to generalize to unseen datasets. We evaluated eight model architectures with publicly available weights on the MESS benchmark.
We present the aggregated results in the following table and refer to the supplementary material of our paper for per-dataset results.


| Model                                                                        |   General |   Earth Monitoring |   Medical Sciences | Engineering |   Agriculture and Biology |   Mean |
|:-----------------------------------------------------------------------------|----------:|-------------------:|-------------------:|------------:|--------------------------:|-------:|
| Random<sup>1</sup>                                                           |      1.17 |               7.11 |              29.51 |       11.71 |                      6.14 |  10.27 |
| Best supervised<sup>2</sup>                                                  |     49.15 |              79.12 |              89.49 |       67.66 |                     81.94 |  71.13 |
| [ZSSeg-B](https://github.com/MendelXu/zsseg.baseline)                        |     19.98 |              17.98 |              41.82 |        14.0 |                     22.32 |  22.73 |
| [ZegFormer-B](https://github.com/dingjiansw101/ZegFormer)                    |     13.57 |              17.25 |              17.47 |       17.92 |                     25.78 |  17.57 |
| [X-Decoder-T](https://github.com/microsoft/X-Decoder)                        |     21.99 |              18.92 |              23.25 |       15.31 |                     19.05 |  19.91 |
| [SAN-B](https://github.com/MendelXu/SAN)                                     |     29.35 |              30.64 |              29.85 |       23.58 |                     15.07 |  26.74 |
| [OpenSeeD-T](https://github.com/IDEA-Research/OpenSeeD)                      |     22.31 |              25.14 |              44.43 |       16.69 |                     10.53 |  24.35 |
| [CAT-Seg-B](https://github.com/KU-CVLAB/CAT-Seg)                             |     34.96 |              34.57 |              41.65 |       26.26 |                     29.32 |  33.74 |
| [Grounded-SAM-B](https://github.com/IDEA-Research/Grounded-Segment-Anything) |     29.51 |              25.97 |              37.38 |       29.51 |                     17.66 |  28.52 |
| [OVSeg-L](https://github.com/facebookresearch/ov-seg)                        |     29.54 |              29.04 |              31.9  |       14.16 |                     28.64 |  26.94 |
| [SAN-L](https://github.com/MendelXu/SAN)                                     |     36.18 |              38.83 |              30.27 |       16.95 |                     20.41 |  30.06 |
| [CAT-Seg-L](https://github.com/KU-CVLAB/CAT-Seg)                             |     39.93 |              39.85 |              48.49 |       26.04 |                     34.06 |  38.14 |
| [Grounded-SAM-L](https://github.com/IDEA-Research/Grounded-Segment-Anything) |     30.32 |              26.44 |              38.69 |       29.25 |                     17.73 |  29.05 |
| [CAT-Seg-H](https://github.com/KU-CVLAB/CAT-Seg)                             |     37.98 |              37.74 |              34.65 |       29.04 |                     37.76 |  35.66 |
| [Grounded-SAM-H](https://github.com/IDEA-Research/Grounded-Segment-Anything) |     30.27 |              26.44 |              38.45 |       28.16 |                     17.67 |  28.78 |

<sup>1</sup> Random is a lower bound. The values represent the expected mean IoU with uniform class distribution.

<sup>2</sup> Supervised are recent supervised models for each dataset individually. We refer to our paper for the details.

### Model adaptions

We provide the adapted code for the evaluated models in separate repositories (currently work in progress):

- ZSSeg: https://github.com/blumenstiel/zsseg.baseline-MESS
- ZegFormer: https://github.com/blumenstiel/ZegFormer-MESS
- OVSeg: https://github.com/blumenstiel/ov-seg-MESS
- X-Decoder: https://github.com/blumenstiel/X-Decoder-MESS
- SAN: https://github.com/blumenstiel/SAN-MESS
- OpenSeeD: https://github.com/blumenstiel/OpenSeeD-MESS
- CAT-Seg: https://github.com/blumenstiel/CAT-Seg-MESS
- Grounded-SAM: https://github.com/blumenstiel/Grounded-SAM-MESS

We also evaluated SAM in a point-to-mask and box-to-mask setting using oracle prompts and provide the Code at https://github.com/blumenstiel/SAM-MESS.

## TODOs

- [ ] Add general evaluation code
- [ ] Publish code of evaluated models   
- [ ] Add preprocessing code for training sets to enable few-shot and supervised settings

## License

The code is released under the [MIT License](LICENSE). The evaluated datasets are released under their respective licenses, see [DATASETS.md](mess/DATASETS.md) for details. Most datasets are limited to non-commercial use only and require a citation which are provided in [mess/datasets.bib](mess/datasets.bib).

## Citation

If you use the MESS benchmark, please cite our paper:

```
@article{MESSBenchmark2023,
  title={{What a MESS: Multi-domain Evaluation of Zero-shot Semantic Segmentation}},
  author={Blumenstiel, Benedikt and Jakubik, Johannes and Kühne, Hilde and Vössing, Michael},
  year={2023}
}
``` 

## Acknowledgement

We would like to acknowledge the work of the dataset providers, especially for the careful collection and annotation of the datasets. Thank you for making the dataset publicly available!
See [DATASETS.md](mess/DATASETS.md) for more details and links to the datasets. We like to further thank the authors of the evaluated models for their work and providing the model weights.