import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

from model_resnet1d import BasicBlock1D, ResNet1D, FCOutputModule #network.model_resnet for running in python

# helper 

def orthogonal_input(x, dim=-1):
    return torch.concatenate((-x[..., 1,:].unsqueeze(-2), x[..., 0,:].unsqueeze(-2)), dim=dim)

## equivariant modules

# class VNLinear(nn.Module):
#     def __init__(
#         self,
#         dim_in,
#         dim_out,
#         scalar_dim_in,
#         scalar_dim_out,
#     ):
#         super().__init__()

#         self.vector_linear = nn.Linear(dim_in, dim_out, bias=False)
#         self.scalar_linear = nn.Linear(scalar_dim_in, scalar_dim_out, bias=False)

#     def forward(self, vector, scalar):
#         return self.vector_linear(torch.concatenate((vector, orthogonal_input(vector, dim=-2)), dim=-1)), self.scalar_linear(scalar)

class VNLinear(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
    ):
        super().__init__()

        self.vector_linear = nn.Linear(dim_in, dim_out, bias=False)

    def forward(self, vector):
        return self.vector_linear(torch.concatenate((vector, orthogonal_input(vector, dim=-2)), dim=-1))
    
class NonLinearity(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        scalar_dim_in,
        scalar_dim_out,
    ):
        super().__init__()
        self.scalar_dim_out = scalar_dim_out
        self.dim_out = dim_out

        self.linear = nn.Linear(dim_in+scalar_dim_in, dim_out+scalar_dim_out, bias=False)
        self.layer_norm = LayerNorm(dim_out+scalar_dim_out)

        
    def forward(self, vector, scalar):
        x = torch.concatenate((torch.norm(vector, dim=-2), scalar), dim=-1)
        x = self.linear(x)
        x = nn.ReLU()(x)
        x = self.layer_norm(x)
        if self.scalar_dim_out == 0:
            return x[..., :self.dim_out].unsqueeze(-2) * (vector/torch.norm(vector, dim=-2).clamp(min = 1e-6).unsqueeze(-2))
        return x[..., :self.dim_out].unsqueeze(-2) * (vector/torch.norm(vector, dim=-2).clamp(min = 1e-6).unsqueeze(-2)), x[..., -self.scalar_dim_out:]
    
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)
    
class VNLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-6): # dim is the vector dimension (i.e., 2)
        super().__init__()
        self.eps = eps
        self.ln = LayerNorm(dim)

    def forward(self, x):
        norms = x.norm(dim = -2)
        x = x / norms.clamp(min = self.eps).unsqueeze(-2)
        return x * self.ln(norms).unsqueeze(-2)
    
class MeanPooling_layer(nn.Module):
    def __init__(
        self,
        dim = 1
    ):
        super().__init__()
        self.dim = dim
        
    def forward(self, vector, scalar):
        return torch.mean(vector, dim=self.dim), torch.mean(scalar, dim=self.dim)
    
class Convolutional(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        scalar_dim_out,
        scalar_dim_in,
        stride = 1,
        padding='same',
        kernel=(16,1),
        bias = False
    ):
        super().__init__()
        self.conv_layer_vec = nn.Conv2d(in_channels=dim_in, out_channels=dim_out, stride=stride, kernel_size=kernel, padding=padding, bias=bias, padding_mode='replicate')
        self.conv_layer_sca = nn.Conv2d(in_channels=scalar_dim_in, out_channels=scalar_dim_out, stride=stride, kernel_size=kernel, padding=padding, bias=bias, padding_mode='replicate')

        
    def forward(self, vector, scalar):
        return self.conv_layer_vec(torch.concatenate((vector, orthogonal_input(vector, dim=-2)), dim=-1).permute(0,3,1,2)).permute(0,2,3,1), self.conv_layer_sca(scalar.unsqueeze(-2).permute(0,3,1,2)).permute(0,2,3,1).squeeze(-2)
    

class Eq_Motion_Model_2vec(nn.Module): ## input vector and scalar separately
    def __init__(
        self,
        dim_in,
        dim_out,
        scalar_dim_in,
        pooling_dim, 
        hidden_dim ,  
        scalar_hidden_dim,
        depth,
        ronin_in_dim,
        ronin_out_dim,
        ronin_depths = [2,2,2,2], 
        ronin_base_plane=64,
        ronin_kernel=3,
        stride = 1,
        padding='same',
        kernel=(16,1),
        bias = False
    ):
        super().__init__()

        self.vnlinear_layer0 = VNLinear(dim_in=2*dim_in, dim_out= hidden_dim)
        self.slinear_layer0 = nn.Linear(scalar_dim_in,scalar_hidden_dim, bias=False)
        self.nonlinearity0 = NonLinearity(dim_in=hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([ 
                Convolutional(dim_in=2*hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim, stride = stride, padding=padding, kernel = kernel, bias=bias),
                NonLinearity(dim_in=hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim),
                VNLayerNorm(hidden_dim), ##for vector
                LayerNorm(scalar_hidden_dim), ## for scalar
            ]))


        self.pooling_layer1 = MeanPooling_layer(dim = pooling_dim)

        ## MLP- linear, nonlinearity, linear
        self.vnlinear_layer1 = VNLinear(dim_in=2*hidden_dim, dim_out= hidden_dim)
        self.slinear_layer1 = nn.Linear(scalar_hidden_dim,scalar_hidden_dim, bias=False)
        self.nonlinearity1 = NonLinearity(dim_in=hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=0)
        self.vnlinear_layer2 = VNLinear(dim_in=2*hidden_dim, dim_out= hidden_dim)

        ##layer normalization
        self.vector_ln1 = VNLayerNorm(hidden_dim) ##for vector
        
        ## output layer
        self.vnoutput_layer = VNLinear(dim_in=2*hidden_dim, dim_out= dim_out)

        ## Ronin
        _fc_config = {'fc_dim': 512, 'in_dim': 7, 'dropout': 0.5, 'trans_planes': 128}
        self.ronin = ResNet1D(ronin_in_dim, ronin_out_dim, BasicBlock1D, ronin_depths, ronin_base_plane, output_block=FCOutputModule, kernel_size=ronin_kernel, **_fc_config)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, vector, scalar, original_scalars):
        v = torch.clone(vector)
        s = torch.clone(scalar)
        v = self.vnlinear_layer0(v)
        s = self.slinear_layer0(s)
        v,s = self.nonlinearity0(v,s)

        ## conv blocks
        for conv, nl, vnln, sln in self.layers:
            v,s = conv(v,s)
            v,s = nl(v,s)
            v = vnln(v)
            s = sln(s)

        v,s = self.pooling_layer1(v,s)

        v = self.vnlinear_layer1(v)
        s = self.slinear_layer1(s)
        v = self.nonlinearity1(v,s) ## no scalar
        v = self.vnlinear_layer2(v)

        ##replace later with batch norm
        v = self.vector_ln1(v)

        v = self.vnoutput_layer(v)

        v1 = v[...,0]/torch.norm(v[...,0], dim=-1,keepdim=True).clamp(min = 1e-8) #(B,d)
        v2 = v[...,1] - (v[...,1].unsqueeze(-2)@v1.unsqueeze(-1)).squeeze(-1)*v1

        #v2 = v[...,[1]] - (v[...,1].unsqueeze(-2)@v1)*v1
        v2 = v2/torch.norm(v2, dim=-1,keepdim=True).clamp(min = 1e-8) #(B,d)
        frame = torch.stack([v1,v2],dim=-1)
        
        frame = frame.permute(0,2,1)
        v = torch.matmul(frame.unsqueeze(1),vector)

        input = torch.concat([v.reshape((*v.shape[:2],-1)), original_scalars], dim=-1).permute(0,2,1)

        vel = self.ronin(input)

        vel = torch.matmul(frame.permute(0,2,1), vel[:,:2].unsqueeze(-1)).squeeze(-1)

        return frame, vel #, disp_inv

if __name__ == "__main__": ## to quickly check
    x = torch.randn((2,200,2,4))
    res = orthogonal_input(x,dim=-2)
    print(res.shape)


    ##-------------------------------------------------------------------
    ## testing the linear layer----------------
    vector = torch.randn((2,200,2,2))
    scalar = torch.randn((2,200,11))
    model_v = VNLinear(dim_in=2*vector.shape[-2], dim_out= 10)
    model_s = nn.Linear(scalar.shape[-1], 15, bias=False)
    ## pass the original values to the model
    out_vec = model_v(vector)
    out_sca = model_s(scalar)

    yaw = torch.randn((1))
    R = torch.Tensor([[torch.cos(yaw), -torch.sin(yaw)], [torch.sin(yaw), torch.cos(yaw)]])

    rotated_vec = einsum('... b c, b a -> ... a c', vector, R)
    out_rot_vec = model_v(rotated_vec)
    out_rot_sca = model_s(scalar)

    rot_out_vec = einsum('... b c, b a -> ... a c', out_vec, R)

    assert torch.allclose(rot_out_vec, out_rot_vec, atol = 1e-6), 'vector is not equivariant'
    assert torch.allclose(out_sca, out_rot_sca, atol= 1e-6), 'scalar is not invariant'
    print('Linear layer test completed!')

    ###------------------------------------------- end of linear layer test

    ##------------------------------------------------------------------
    ## testing non-linearity
    out_vec = torch.randn((2,200,2,10))
    out_sca = torch.randn((2,200,15))
    rot_out_vec = einsum('... b c, b a -> ... a c', out_vec, R)
    nonlinear_layer = NonLinearity(dim_in=out_vec.shape[-1], dim_out= 10, scalar_dim_in=out_sca.shape[-1], scalar_dim_out=20)
    nl_out_vec, nl_out_sca = nonlinear_layer(out_vec, out_sca)

    nl_out_rot_out_vec, rot_nl_out_sca = nonlinear_layer(rot_out_vec, out_sca)

    rot_nl_out_vec = einsum('... b c, b a -> ... a c', nl_out_vec, R)

    assert torch.allclose(rot_nl_out_vec, nl_out_rot_out_vec, atol = 1e-6), 'vector is not equivariant'
    assert torch.allclose(nl_out_sca, rot_nl_out_sca, atol= 1e-6), 'scalar is not invariant'
    print('Non linearity layer test completed!')

    ###------------------------------------------- end of non linearity layer test

    ##---------------------------------------------------------------------------------
    ## checking the layer norm

    vector = torch.randn((32,200,2,20))
    scalar = torch.randn((32,200,14))

    vec_ln_layer = VNLayerNorm(dim=vector.shape[-1])

    ## pass the original values to the model
    out_vec = vec_ln_layer(vector)

    sca_ln_layer = LayerNorm(dim=scalar.shape[-1])

    out_sca = sca_ln_layer(scalar)

    yaw = torch.randn((1))
    R = torch.Tensor([[torch.cos(yaw), -torch.sin(yaw)], [torch.sin(yaw), torch.cos(yaw)]])


    rotated_vec = einsum('... b c, b a -> ... a c', vector, R)
    out_rot_vec = vec_ln_layer(rotated_vec)

    rot_out_vec = einsum('... b c, b a -> ... a c', out_vec, R)

    assert torch.allclose(rot_out_vec, out_rot_vec, atol = 1e-6), 'vector is not equivariant'
    assert torch.allclose(out_sca, out_sca, atol= 1e-6), 'scalar is not invariant'
    print('Layer norm test completed!')
    ###------------------------------------------- end of layer norm

    ##-----------------------------------------------------------------
    ##testing Mean Pooling
    vector = torch.randn((2,200,2,10))
    scalar = torch.randn((2,200,11))
    pooling_layer = MeanPooling_layer(dim = 1) ## over the number of samples
    ## pass the original values to the model
    out_vec, out_sca = pooling_layer(vector, scalar)

    yaw = torch.randn((1))
    R = torch.Tensor([[torch.cos(yaw), -torch.sin(yaw)], [torch.sin(yaw), torch.cos(yaw)]])

    rotated_vec = einsum('... b c, b a -> ... a c', vector, R)
    out_rot_vec, out_rot_sca = pooling_layer(rotated_vec, scalar)

    rot_out_vec = einsum('... b c, b a -> ... a c', out_vec, R)

    assert torch.allclose(rot_out_vec, out_rot_vec, atol = 1e-6), 'vector is not equivariant'
    assert torch.allclose(out_sca, out_rot_sca, atol= 1e-6), 'scalar is not invariant'
    print('Mean Pooling layer test completed!')

    ##-------------------------------------------- end of mean pooling layer test

    ##------------------------------------------------------------------
    ## testing convolutional 

    vector = torch.randn((2,200,2,4))
    scalar = torch.randn((2,200,14))
    rot_vec = einsum('... b c, b a -> ... a c', vector, R)
    conv_layer = Convolutional(dim_in=2*vector.shape[-1], dim_out= 10, scalar_dim_in=scalar.shape[-1], scalar_dim_out=15, stride=1, padding="same", kernel = (16,1), bias=False)
    ## pass the original values to the model
    out_vec, out_sca = conv_layer(vector, scalar)

    out_rot_vec, out_rot_sca = conv_layer(rot_vec, scalar)

    rot_out_vec = einsum('... b c, b a -> ... a c', out_vec, R)

    assert torch.allclose(rot_out_vec, out_rot_vec, atol = 1e-6), 'vector is not equivariant'
    assert torch.allclose(out_sca, out_rot_sca, atol= 1e-6), 'scalar is not invariant'
    print('Convolutional layer test completed!')

    ###------------------------------------------- end of convolutional test


    ##---------------------------------------------------------------------------------
    # checking the full equivariant model

 
    vector = torch.randn((2,200,2,2))
    scalar = torch.randn((2,200,5))
    original_scalars = torch.randn((2,200,2))

    
    eq_model = Eq_Motion_Model_2vec(dim_in=2, dim_out= 2, scalar_dim_in=5, pooling_dim = 1,
                               ronin_in_dim=6, ronin_out_dim=2,hidden_dim=128, scalar_hidden_dim=128, depth=1, 
                               stride = 1, padding='same', kernel =(32,1), bias=False
                            )

    eq_model.eval()
    total_params = sum(p.numel() for p in eq_model.parameters() if p.requires_grad)
    print('Network eq frame 2 scalars loaded to device ')
    print("Total number of parameters:", total_params)

    ## pass the original values to the model
    frame, out_vel = eq_model(vector, scalar, original_scalars)#, d_inv

    yaw = torch.randn((1))
    R = torch.Tensor([[torch.cos(yaw), -torch.sin(yaw)], [torch.sin(yaw), torch.cos(yaw)]])

    rotated_vec = vector.permute(0,1,3,2) @ R
    rotated_vec = rotated_vec.permute(0,1,3,2)
    # einsum('... b c, b a -> ... a c', vector, R)
    frame,out_rot_d = eq_model(rotated_vec, scalar, original_scalars) #, d_inv

    # rot_out_vec = einsum('... b c, b a -> ... a c', out_vec, R)
    R_o = torch.Tensor([[torch.cos(yaw), torch.sin(yaw)], [-torch.sin(yaw), torch.cos(yaw)]])
    rot_out_d = torch.matmul(R_o.unsqueeze(0),out_vel.unsqueeze(-1)).squeeze(-1)

    assert torch.allclose(rot_out_d, out_rot_d, atol = 1e-4), 'vector is not equivariant'

    print('Full equivariant model test completed!')

    ###------------------------------------------- end of full equivariant model test

 
   