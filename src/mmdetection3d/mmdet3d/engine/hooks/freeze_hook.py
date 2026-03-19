from mmengine.hooks import Hook
from mmengine.logging import print_log

from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class FreezeLayersBeforeTrainHook(Hook):
    def __init__(self, layer_names):
        self.layer_names = layer_names

    def before_train(self, runner):
        print_log('Start freezing layers.', logger='current')
        model = runner.model
        
        if hasattr(model, 'module'):
            model = model.module
            
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in self.layer_names):
                param.requires_grad = False
                print_log(f'Freezing layer: {name}.', logger='current')
                
        for name, param in model.named_parameters():
            print_log(f'Layer {name}: requires_grad = {param.requires_grad}', logger='current')