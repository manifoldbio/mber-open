from mber.models.folding.folding_model_bases import ProteinFoldingModel
from mber.models.folding.nbb2_model import NBB2Model
from mber.models.folding.abb2_model import ABB2Model
from mber.models.folding.esmfold_model import ESMFoldModel
# from .af2_model import AF2Model

FOLDING_MODELS = {
    'nbb2': NBB2Model,
    'abb2': ABB2Model,
    'esmfold': ESMFoldModel,
    # 'af2': AF2Model,
}