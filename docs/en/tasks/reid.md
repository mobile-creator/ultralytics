---
comments: true
description: Learn about person re-identification (ReID) using YOLO26. Train, validate, predict, and export ReID models for matching people across camera views.
keywords: YOLO26, person re-identification, ReID, metric learning, Market-1501, embedding extraction, train, validate, predict
model_name: yolo26n-reid
---

# Person Re-Identification (ReID)

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/person-reid-overview.avif" alt="YOLO person re-identification matching people across camera views">

Person re-identification (ReID) is the task of matching the same individual across different camera views or time instances. Unlike object detection which locates objects, or classification which categorizes images, ReID produces a compact embedding vector for each person image that can be compared against other embeddings to determine identity matches.

The output of a ReID model is a fixed-dimensional embedding vector. Two images of the same person should produce embeddings that are close in distance, while images of different people should produce embeddings that are far apart.

!!! tip

    YOLO26 ReID models use the `-reid` suffix, i.e., `yolo26n-reid.pt`, and support [Market-1501](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/Market-1501.yaml), [DukeMTMC-reID](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/DukeMTMC-reID.yaml), and [MSMT17](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/MSMT17.yaml) datasets.

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26)

YOLO26 ReID models are available in multiple sizes. All models use a BNNeck architecture with PK batch sampling and multi-loss training (cross-entropy + triplet + optional center/supcon losses).

| Model                                                                                    | size<br><sup>(pixels) | mAP<br><sup>Market-1501 | Rank-1<br><sup>Market-1501 | mAP<br><sup>DukeMTMC | Rank-1<br><sup>DukeMTMC | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------------------------------------------------------------------------------------- | --------------------- | ----------------------- | -------------------------- | --------------------- | ------------------------ | ------------------- | ------------------ |
| [YOLO26n-reid](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26n-reid.pt) | 256                   | 23.7                    | 42.5                       | 16.4                  | 30.5                     | 2.0                 | 3.3                |
| [YOLO26s-reid](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26s-reid.pt) | 256                   | 29.4                    | 50.4                       | 16.9                  | 30.7                     | 6.5                 | 12.7               |

- **mAP** and **Rank-1** values are on the [Market-1501](../datasets/reid/market1501.md) and [DukeMTMC-reID](../datasets/reid/dukemtmc.md) datasets (60 epochs, SGD, imgsz=256). <br>Reproduce by `yolo reid val data=Market-1501.yaml device=0`
- **Params** and **FLOPs** values are for the fused model after `model.fuse()`.

## Train

Train a YOLO26n-reid model on the Market-1501 dataset for 60 epochs at image size 256. For a full list of available arguments see the [Configuration](../usage/cfg.md) page.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-reid.yaml")  # build a new model from YAML
        model = YOLO("yolo26n-reid.pt")  # load a pretrained model (recommended for training)
        model = YOLO("yolo26n-reid.yaml").load("yolo26n-cls.pt")  # build from YAML and transfer weights

        # Train the model
        results = model.train(data="Market-1501.yaml", epochs=60, imgsz=256)
        ```

    === "CLI"

        ```bash
        # Build a new model from YAML and start training from scratch
        yolo reid train data=Market-1501.yaml model=yolo26n-reid.yaml epochs=60 imgsz=256

        # Start training from a pretrained *.pt model
        yolo reid train data=Market-1501.yaml model=yolo26n-reid.pt epochs=60 imgsz=256

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo reid train data=Market-1501.yaml model=yolo26n-reid.yaml pretrained=yolo26n-cls.pt epochs=60 imgsz=256
        ```

### ReID-specific training arguments

| Argument         | Default | Description                                                     |
| ---------------- | ------- | --------------------------------------------------------------- |
| `reid_p`         | `16`    | Number of identities per batch (P in PK sampling)               |
| `reid_k`         | `4`     | Number of images per identity (K in PK sampling)                |
| `triplet_margin` | `0.3`   | Margin for batch-hard triplet loss                              |
| `triplet_weight` | `1.0`   | Weight for triplet loss                                         |
| `ce_weight`      | `1.0`   | Weight for cross-entropy identity classification loss           |
| `center_weight`  | `0.0`   | Weight for center loss (0 = disabled)                           |
| `center_momentum`| `0.9`   | EMA momentum for center loss class centers                      |
| `focal_gamma`    | `0.0`   | Focal loss gamma for ReID CE loss (0 = standard CE)             |
| `supcon_temp`    | `0.0`   | Supervised contrastive loss temperature (0 = use triplet loss)  |

!!! tip

    The effective batch size is `reid_p * reid_k`. For better hard-negative mining, use larger `reid_k` values (e.g., `reid_k=8` with `reid_p=32` for batch size 256).

### Dataset format

YOLO ReID dataset format can be found in detail in the [Dataset Guide](../datasets/reid/index.md).

## Val

Validate a trained YOLO26n-reid model on the Market-1501 dataset. The evaluation uses the standard Market-1501 protocol: L2 distance between query and gallery embeddings, excluding same-pid-same-camid matches. Reports mAP and Rank-1/5/10 accuracy.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-reid.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        metrics.results_dict["metrics/mAP"]  # mAP
        metrics.results_dict["metrics/rank1"]  # Rank-1 accuracy
        ```

    === "CLI"

        ```bash
        yolo reid val model=yolo26n-reid.pt  # val official model
        yolo reid val model=path/to/best.pt  # val custom model
        ```

### Test-Time Augmentation (TTA)

Enable horizontal flip TTA with `reid_tta=True` to average embeddings from original and horizontally-flipped images. This typically improves mAP by 1-2% at the cost of doubling inference time.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("path/to/best.pt")
        metrics = model.val(reid_tta=True)
        ```

    === "CLI"

        ```bash
        yolo reid val model=path/to/best.pt reid_tta=True
        ```

### K-Reciprocal Re-Ranking

Enable [k-reciprocal re-ranking](https://arxiv.org/abs/1701.08398) with `reid_reranking=True` to refine distance rankings using neighborhood structure. This post-processing technique (Zhong et al., CVPR 2017) can improve mAP by **15-17%** with no additional training — it only modifies the distance matrix at evaluation time. Re-ranking increases evaluation time due to the additional pairwise computations.

For best results, combine both TTA and re-ranking:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("path/to/best.pt")
        metrics = model.val(reid_tta=True, reid_reranking=True)
        ```

    === "CLI"

        ```bash
        yolo reid val model=path/to/best.pt reid_tta=True reid_reranking=True
        ```

### ReID evaluation arguments

| Argument         | Default | Description                                                                  |
| ---------------- | ------- | ---------------------------------------------------------------------------- |
| `reid_tta`       | `False` | Enable horizontal flip TTA (+1-2% mAP, 2x inference time)                   |
| `reid_reranking` | `False` | Enable k-reciprocal re-ranking (+15-17% mAP, increases eval time)            |

## Predict

Use a trained YOLO26n-reid model to extract embedding vectors from person images.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-reid.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Predict with the model (extract embeddings)
        results = model("path/to/person.jpg")
        ```

    === "CLI"

        ```bash
        yolo reid predict model=yolo26n-reid.pt source='path/to/person.jpg'  # predict with official model
        yolo reid predict model=path/to/best.pt source='path/to/person.jpg'  # predict with custom model
        ```

See full `predict` mode details in the [Predict](../modes/predict.md) page.

## Export

Export a YOLO26n-reid model to a different format like ONNX, CoreML, etc.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-reid.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom-trained model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n-reid.pt format=onnx  # export official model
        yolo export model=path/to/best.pt format=onnx   # export custom-trained model
        ```

Available YOLO26-reid export formats are in the table below. You can export to any format using the `format` argument, i.e., `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e., `yolo predict model=yolo26n-reid.onnx`. Usage examples are shown for your model after export completes.

{% include "macros/export-table.md" %}

See full `export` details in the [Export](../modes/export.md) page.

## FAQ

### What is person re-identification (ReID) and how does YOLO26 handle it?

Person re-identification (ReID) is the task of recognizing the same person across different camera views or at different times. YOLO26 ReID models produce compact embedding vectors from person images. These embeddings can be compared using distance metrics (e.g., L2 or cosine distance) to determine if two images show the same person. The model is trained with PK batch sampling and a combination of cross-entropy, triplet, and optional center/supervised contrastive losses for robust metric learning.

### How do I train a YOLO26 ReID model?

To train a YOLO26 ReID model on Market-1501:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n-reid.yaml")
        results = model.train(data="Market-1501.yaml", epochs=60, imgsz=256)
        ```

    === "CLI"

        ```bash
        yolo reid train data=Market-1501.yaml model=yolo26n-reid.yaml epochs=60 imgsz=256
        ```

Key hyperparameters include `reid_p` (identities per batch), `reid_k` (images per identity), `triplet_margin`, and loss weights. For more details, see the [Configuration](../usage/cfg.md) page.

### What datasets are supported for ReID training?

The following datasets are supported out of the box:

| Dataset | Train Images | IDs | Cameras | Config |
|---------|-------------|-----|---------|--------|
| [Market-1501](../datasets/reid/market1501.md) | 12,936 | 751 | 6 | `Market-1501.yaml` |
| [DukeMTMC-reID](../datasets/reid/dukemtmc.md) | 16,522 | 702 | 8 | `DukeMTMC-reID.yaml` |
| [MSMT17](../datasets/reid/msmt17.md) | 30,248 | 1,041 | 15 | `MSMT17.yaml` |

Custom datasets are also supported via configurable filename regex patterns. See the [ReID Datasets](../datasets/reid/index.md) guide for format details.

### What is PK batch sampling in ReID training?

PK sampling is a batch construction strategy where each batch contains P randomly selected identities, each with K randomly sampled images. This ensures every batch has multiple images per identity, which is essential for computing meaningful triplet losses that require positive pairs (same identity) and negative pairs (different identities) within each batch.

### How can I improve ReID evaluation accuracy without retraining?

Two post-processing techniques are available that improve mAP at evaluation time with no retraining needed:

1. **Flip TTA** (`reid_tta=True`): Averages embeddings from original and horizontally-flipped images. Typically +1-2% mAP.
2. **K-reciprocal re-ranking** (`reid_reranking=True`): Refines distance rankings using neighborhood structure ([Zhong et al., CVPR 2017](https://arxiv.org/abs/1701.08398)). Typically +15-17% mAP.

```bash
yolo reid val model=path/to/best.pt reid_tta=True reid_reranking=True
```

### How does ReID differ from image classification?

While both tasks use backbone networks for feature extraction, they solve different problems:

- **Classification** assigns one of N predefined labels to an image. The model outputs class probabilities.
- **ReID** produces an embedding vector that captures identity-discriminative features. The model must generalize to identities never seen during training, making it an open-set problem. At inference time, embeddings are compared by distance rather than classified.
