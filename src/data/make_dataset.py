# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import numpy as np
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    imgs = []
    labels = []
    for i in range(5):
        train = np.load(f"data/raw/train_{i}.npz")
        images_ = train["images"]
        labels_ = train["labels"]
        imgs.append(images_)
        labels.append(labels_)
    images = np.concatenate(imgs)
    labels = np.concatenate(labels)

    # transform = T.Compose([T.ToTensor(),
    #                        T.Normalize([0], [1])])
    # images_t = transform(images)

    images_t = (images - images.mean()) / images.std()

    np.savez("data/processed/train.npz", images=images_t, labels=labels)

    test = np.load("data/raw/test.npz")
    images = test["images"]
    labels = test["labels"]
    images_t = (images - images.mean()) / images.std()

    np.savez("data/processed/test.npz", images=images_t, labels=labels)

    np.savez(
        "data/interim/example_images.npz", images=images_t[0:10], labels=labels[0:10]
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
