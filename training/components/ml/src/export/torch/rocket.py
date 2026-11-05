import torch
import torch.nn as nn
import torch.nn.functional as F

class Rocket(nn.Module):

    def __init__(self,  
                 normalize: bool,
                 weights : torch.Tensor,            # size: (length[i] * num_channel_indices[i]).sum()
                 lengths : torch.Tensor,            # size: num_kernels
                 biases: torch.Tensor,              # size: num_kernels
                 dilations: torch.Tensor,           # size: num_kernels
                 paddings: torch.Tensor,            # size: num_kernels
                 num_channel_indices: torch.Tensor, # size: num_kernels (how many channels each kernel considers)
                 channel_indices: torch.Tensor,     # size: num_channel_indices.sum() (channel index for each one of the kernels, flattened)            
                 eps: float = 1e-10):
        super().__init__()

        self.normalize = normalize
        windices = torch.cat((torch.zeros((1,),dtype=torch.int64), torch.cumsum(lengths * num_channel_indices,dim=0)))
        cindices = torch.cat((torch.zeros((1,),dtype=torch.int64), torch.cumsum(num_channel_indices,dim=0)))
        self.register_buffer('weights',weights.float())
        self.register_buffer('lengths',lengths)
        self.register_buffer('biases',biases.float())
        self.register_buffer('dilations',dilations)
        self.register_buffer('paddings',paddings)
        self.register_buffer('num_channel_indices',num_channel_indices.long())
        self.register_buffer('channel_indices',channel_indices.long())
        self.register_buffer('windices',windices)
        self.register_buffer('cindices',cindices)
        self.num_kernels = len(lengths)
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if self.normalize:
            std_, mean_ = torch.std_mean(input = x, keepdim=True, dim = -1)            
            x = (x - mean_) / (std_ + self.eps)
            
        # feature extraction
        num_kernels = self.num_kernels
        weights, lengths, biases, dilations, paddings, num_channel_indices, channel_indices = \
            self.weights, self.lengths, self.biases, self.dilations,self.paddings, self.num_channel_indices, self.channel_indices

        windices = self.windices
        cindices = self.cindices

        y = torch.zeros((x.shape[0],2*num_kernels), device = x.device)
        for k in range(num_kernels):

            w = weights[windices[k]:windices[k+1]].reshape(num_channel_indices[k],lengths[k])
            b = biases[k]
            dilation = int(dilations[k].item())
            pad = int(paddings[k].item())

            pyx = x[:,channel_indices[cindices[k]:cindices[k+1]],:].float()

            pyy = F.conv1d(pyx,w.unsqueeze(0),torch.tensor([b],device=b.device),1,pad,dilation)
            ppv = ((pyy > 0).sum(dim=2) / pyy.shape[2]).squeeze()
            #mx = pyy.amax(dim=2).squeeze()
            mx = ((F.softmax(pyy*1000,dim=2))*pyy).sum(dim=2).squeeze()       # ONNX does not support amax
           
            y[:,2*k] = ppv
            y[:,2*k+1] = mx        

        return y