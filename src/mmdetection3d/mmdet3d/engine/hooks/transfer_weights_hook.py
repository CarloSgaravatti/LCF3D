import torch
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.runner.checkpoint import _load_checkpoint_with_prefix, load_state_dict

from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class LayerWeightTransferHook(Hook):
    '''
    This class transfer the weights from a pretrained model when the module does not
    provide the init_cfg parameter for some reason.
    '''
    def __init__(self, checkpoint_path, layer_pairs):
        """
        Initialize the LayerWeightTransferHook.

        Args:
            checkpoint_path (str): Path to the original checkpoint file.
            layer_pairs (list of tuples): Each tuple contains two elements:
                                          (layer_name in original network, target_layer_name in new network).
        """
        self.checkpoint_path = checkpoint_path
        self.layer_pairs = layer_pairs

    def before_train(self, runner):
        """
        Called before training starts.

        Args:
            runner (mmengine.Runner): The runner that manages the training process.
        """
        model = runner.model
        
        if hasattr(model, 'module'):
            model = model.module
            
        model = dict(model.named_modules())['']

        for orig_layer_name, target_layer_name in self.layer_pairs:
            
            target_layer = dict(model.named_modules()).get(target_layer_name, None)
            if target_layer is None:
                runner.logger.warning(f"Layer {target_layer_name} not found in the target model.")
                continue
            
            state_dict = _load_checkpoint_with_prefix(
                orig_layer_name, self.checkpoint_path)
            load_state_dict(target_layer, state_dict, strict=False, logger='current')
            runner.logger.info(f"Transferred weights from {orig_layer_name} to {target_layer_name}.")

        runner.logger.info("Weight transfer completed.")