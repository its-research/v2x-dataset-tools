"""Datasets Arguments."""
from dataclasses import dataclass

from simple_parsing import choice, field
from simple_parsing.helpers import flag


@dataclass
class DatasetsArguments:
    """Datasets Arguments."""

    dataset_type: str = choice(
        'V2X-Seq-SPD',
        'V2X-Seq-TFD',
        'DAIR-V2X-I',
        'DAIR-V2X-V',
        'DAIR-V2X-C',
        'ROPE3D',
        default='DAIR-V2X-C',
    )
    show: bool = flag(default=False)
    save: bool = flag(default=False)
    save_path: str = field(default='outputs/')
