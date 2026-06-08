from giga_train import TRANSFORMS

from .wm_rnd_mix_transforms import WMRNDMixTransforms


@TRANSFORMS.register
class WMRNDMixStageTwoTransforms(WMRNDMixTransforms):
    """Stage-two transform for RND-mix.

    Stage-two 3D conditions are generated online in trainer rollout,
    so this transform reuses stage-one RGB+depth preprocessing only.
    """

    pass
