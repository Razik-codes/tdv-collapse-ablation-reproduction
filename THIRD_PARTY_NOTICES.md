# Source and third-party notices

This repository combines an independent reproduction artifact with the original TDV research code and adapted third-party implementations. This file makes that lineage explicit; it does not replace the license files distributed with each component.

## Reproduction artifact

The repository-level additions centered on [`claim_tutorial.py`](claim_tutorial.py) and [`reports/collapse-ablation/`](reports/collapse-ablation/) reproduce one ablation from the TDV paper at reduced scale. Their experimental substitutions, branch lineage, compute budget, and limitations are documented in the report.

## Original TDV project

The model, training framework, data loaders, and evaluation integration originate from [Ninad Daithankar and collaborators' TDV project](https://temporal-difference-vision.github.io/), associated with the paper *You Don't Need Strong Assumptions: Visual Representation Learning via Temporal Differences*. The repository root is distributed under the [Apache License 2.0](LICENSE).

## Bundled and adapted projects

| Path | Upstream project | License information |
|---|---|---|
| `repos/dino-with-online-knn/` | [DINO](https://github.com/facebookresearch/dino), with local online-KNN changes | [Apache License 2.0](repos/dino-with-online-knn/LICENSE) |
| `repos/ibot-with-online-knn/` | [iBOT](https://github.com/bytedance/ibot), with local online-KNN changes | [Apache License 2.0](repos/ibot-with-online-knn/LICENSE) |
| `repos/mmsegmentation-tdv/` | [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), adapted with a TDV backbone | [Apache License 2.0](repos/mmsegmentation-tdv/LICENSE) |
| `eval/flow/croco/` | [CroCo v2](https://github.com/naver/croco), adapted for optical-flow and stereo evaluation | [CC BY-NC-SA 4.0 and upstream notices](eval/flow/croco/LICENSE) |
| `model/cv/dinov2/` | Components adapted from [DINOv2](https://github.com/facebookresearch/dinov2) | DINOv2's upstream Apache License 2.0 terms apply |

The CroCo-derived evaluation code is non-commercial under CC BY-NC-SA 4.0. Users are responsible for checking the licenses and terms of datasets, pretrained weights, and external dependencies before reuse or redistribution.
