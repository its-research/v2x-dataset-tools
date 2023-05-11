"""Datasets Arguments"""
from dataclasses import dataclass

from simple_parsing import choice


@dataclass
class DatasetsArguments:
    """Datasets Arguments"""

    dataset_type: str = choice("V2X-Seq-SPD", "V2X-Seq-TFD", "DAIR-V2X-I", "DAIR-V2X-V", "DAIR-V2X-C", default="DAIR-V2X-C")
