from .tranformer_wan_rnd import WanTransformerRNDModel


class WanConditionRNDStageTwoModel(WanTransformerRNDModel):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_depth_condition", True)
        kwargs.setdefault("zero_init_depth_condition", True)
        super().__init__(*args, **kwargs)
