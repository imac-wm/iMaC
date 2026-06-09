from .tranformer_wan_rnd import WanTransformerRNDModel


class WanConditionRNDStageOneModel(WanTransformerRNDModel):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_depth_condition", False)
        super().__init__(*args, **kwargs)
