# Lougat

This project aims to create a text scanner that converts paper images into machine-readable formats (e.g., Markdown, JSON). It is the son of Nougat, and thus, grandson of Douat.

The key idea is to combine the bounding box modality with text, achieving a pixel scan behavior that predicts not only the next token but also the next position.

![Example Image](images/image.png)

The name "Lougat" is a combination of LLama and Nougat. In this repo, you'll also find other combinations like: 

- Florence2 + LLama → Flougat
- Sam2 + LLama → Slougat
- Nougat + Relative Position Embedding LLama → Rlougat

The key idea is nature continues of this paper [LOCR: Location-Guided Transformer for Optical Character Recognition]([[2403.02127\] LOCR: Location-Guided Transformer for Optical Character Recognition (arxiv.org)](https://arxiv.org/abs/2403.02127))

## Dataset

The dataset is the [UltexB](https://huggingface.co/datasets/LLM4SCIENCE/uparxive_boxed_pdf/tree/main) dataset from [Uparxive](https://github.com/veya2ztn/uparxive/blob/release/documents/the_uparxive_text_level_boxed_dataset.md) 

