import torch 
import torch.nn as nn
import torch.nn.functional as F



## lets combine this into function:
def w8_16_forward(weight, input, scale, bias=None):
    casted_weight = weight.to(input.dtype) 
    output = F.linear(input, casted_weight) * scale
    if bias is not None:
        output += bias
    return output


class W8A16LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=torch.float32):
        super().__init__()

        self.register_buffer("int8_weight",
                            torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8))
        
        self.register_buffer("scale", torch.randn((out_features)
                             , dtype=dtype))
        
        if bias:
            self.register_buffer("bias", torch.zeros((1, out_features)
                                 , dtype=dtype))
        else:
            self.bias = None

    def quantize(self, weight):
        ## cast to fp32 is recommended for stability
        w_fp32 = weight.clone().to(torch.float32)

        scales = w_fp32.abs().max(dim=-1).values / 127 
        scales = scales.to(weight.dtype) 

        int8_weight = torch.round(w_fp32 / scales.unsqueeze(1)).to(torch.int8)

        self.int8_weight = int8_weight
        self.scale = scales

    def forward(self, input):
        return w8_16_forward(self.int8_weight, input, self.scale, self.bias)
    
def replace_module_and_quantize(module,
                target_class, 
                module_name_to_exclude):
    
    for name, child in module.named_children():

        # if the layer is a linear layer and not in the exclusion list
        if isinstance(child, nn.Linear) and not any([x == name for x in module_name_to_exclude]):
            old_bias = child.bias
            old_weight = child.weight

            new_module = target_class(
                child.in_features,
                child.out_features,
                child.bias is not None,
                child.weight.dtype
            )
            setattr(module, name, new_module)
            getattr(module , name).quantize(old_weight)

            if old_bias is not None:
                getattr(module , name).bias = old_bias 
        
        else : 
            # recursively call the function
            replace_module_and_quantize(child, target_class, module_name_to_exclude)

