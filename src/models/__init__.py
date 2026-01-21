from .ultra import ULTRA
from .db_exec import DBExecModel
from .multihop import MultiHopPredictor
from .topology import ThreeHopPipeline, TwoUnionPipeline, TwoIntersectProjectPipeline
from .ultra import parse_args as ultra_parse_args
from .ultra import shape_input as ultra_shape_input
from .common import ModelUtils

__all_models__ = [
    "ULTRA",
    "DBExecModel",
    "MultiHopPredictor",
    "ThreeHopPipeline",
    "TwoUnionPipeline",
    "TwoIntersectProjectPipeline"
]

MODEL_DEPENDENCIES = {
    "ULTRA": [],
    "DBExecModel": [],
    "MultiHopPredictor": [],
    "ThreeHopPipeline": [],
    "TwoUnionPipeline": [],
    "TwoIntersectProjectPipeline": []
}

VECTORDB_USAGE = {
    "ULTRA": False,
    "DBExecModel": False,
    "MultiHopPredictor": False,
    "ThreeHopPipeline": False,
    "TwoUnionPipeline": False,
    "TwoIntersectProjectPipeline": False
}

RSPMM_USAGE = {
    "ULTRA": True,
    "DBExecModel": False,
    "MultiHopPredictor": False,
    "ThreeHopPipeline": False,
    "TwoUnionPipeline": False,
    "TwoIntersectProjectPipeline": False
}

NUMNODES_USAGE = {
    "ULTRA": False,
    "DBExecModel": False,
    "MultiHopPredictor": False,
    "ThreeHopPipeline": False,
    "TwoUnionPipeline": False,
    "TwoIntersectProjectPipeline": False
}

HAVE_PRETRAINED_MODEL = {
    "ULTRA": True,
    "DBExecModel": False,
    "MultiHopPredictor": False, 
    "ThreeHopPipeline": False,
    "TwoUnionPipeline": False,
    "TwoIntersectProjectPipeline": False
}
