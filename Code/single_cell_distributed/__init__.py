"""Dedicated multi-GPU training package for the SingleCell cohorts."""

import os


# This flag is set before Main/DataHandler/Model import the shared parameter
# module. It keeps DDP-only CLI options out of the ordinary training entry.
os.environ["HIHYPERDR_SINGLE_CELL_DDP"] = "1"
