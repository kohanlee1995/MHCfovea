import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SaveValues():
    def __init__(self, model):
        # register a hook to save values of activations and gradients
        self.activations = None
        self.gradients = None
        self.forward_hook = model.register_forward_hook(self.get_act)
        self.backward_hook = model.register_backward_hook(self.get_grad)

    def get_act(self, module, input, output):
        self.activations = output

    def get_grad(self, module, input, output):
        self.gradients = output[0]

    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


class CAM(object):
    def __init__(self, model, target_layer, len_tuple, cuda, cat_dim):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.values = SaveValues(self.target_layer)
        self.mhc_len_pre, self.mhc_len_post, self.epitope_len_pre, self.epitope_len_post = len_tuple
        self.cuda = cuda
        self.cat_dim = cat_dim

    def forward(self, mhc, epitope):
        '''
        mhc: shape -> (n, encode_channel, mhc_len)
        epitope: shape -> (n, encode_channel, epitope_len)

        '''
        if self.cuda:
            self.model = self.model.cuda()
            mhc = mhc.cuda()
            epitope = epitope.cuda()
        
        x = self.model(mhc, epitope)
        
        # get fc layer
        for n, m in self.model.main._modules.items(): #specific model architecture
            if isinstance(m, nn.Linear):
                weight_fc = list(m.parameters())[0].data
        
        mhc_cam, epitope_cam = self.getCAM(self.values, weight_fc)
        
        return mhc_cam.cpu().detach().numpy(), epitope_cam.cpu().detach().numpy()

    def __call__(self, mhc, epitope):
        return self.forward(mhc, epitope)

    def getCAM(self, values, weight_fc):
        '''
        values.activations: shape -> (n, C, L)
        weight_fc: shape -> (n, C)
        cam: shape -> (n, 1, L)
        '''
        cam = F.conv1d(values.activations, weight=weight_fc[:, :, None])
        n, _, _ = cam.shape

        # seperate cam
        mhc_cam = cam[:, :, :self.mhc_len_post]
        epitope_cam = cam[:, :, self.mhc_len_post:]
        
        if self.mhc_len_post != 0:
            mhc_cam = F.interpolate(mhc_cam, self.mhc_len_pre, mode="linear")
            mhc_cam = self._min_max_norm(mhc_cam, 2)
            mhc_cam = mhc_cam.view(n, self.mhc_len_pre)
        if self.epitope_len_post != 0:
            epitope_cam = F.interpolate(epitope_cam, self.epitope_len_pre, mode="linear")
            epitope_cam = self._min_max_norm(epitope_cam, 2)
            epitope_cam = epitope_cam.view(n, self.epitope_len_pre)

        return mhc_cam, epitope_cam

    def _min_max_norm(self, tensor, dim):
        tensor_min = tensor.min(dim=dim, keepdim=True)[0]
        tensor_max = tensor.max(dim=dim, keepdim=True)[0]
        denominator = tensor_max - tensor_min
        denominator = denominator.where(denominator != 0, torch.ones_like(tensor_min))
        return (tensor - tensor_min) / denominator


class GradCAM(CAM):
    def __init__(self, model, target_layer, len_tuple, cuda, cat_dim):
        super().__init__(model, target_layer, len_tuple, cuda, cat_dim)

    def forward(self, mhc, epitope):
        '''
        mhc: shape -> (n, encode_channel, mhc_len)
        epitope: shape -> (n, encode_channel, epitope_len)
        '''
        if self.cuda:
            self.model = self.model.cuda()
            mhc = mhc.cuda()
            epitope = epitope.cuda()

        # no sigmoid function
        x = torch.cat((self.model.modelA(mhc), self.model.modelB(epitope)), dim=self.cat_dim) #specific model architecture
        for n, m in self.model.main._modules.items(): #specific model architecture
            if isinstance(m, nn.Sigmoid):
                break
            else:
                x = m(x)

        mhc_cam, epitope_cam = self.getGradCAM(self.values, x)
        
        return mhc_cam.cpu().detach().numpy(), epitope_cam.cpu().detach().numpy()

    def __call__(self, mhc, epitope):
        return self.forward(mhc, epitope)

    def getGradCAM(self, values, x):
        '''
        values.activations: shape -> (n, C, L)
        values.gradients: shape -> (n, C, L)
        x: shape -> (n, 1)
        alpha: shape -> (n, C, 1)
        cam: shape -> (n, 1, L)
        '''
        n, _ = x.shape

        self.model.zero_grad()
        if self.cuda:
            x.backward(torch.ones(n, 1).cuda(), retain_graph=True)
        else:
            x.backward(torch.ones(n, 1), retain_graph=True)

        # calculate alpha
        alpha = values.gradients.mean(dim=2, keepdim=True)

        cam = (alpha * values.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # seperate cam
        mhc_cam = cam[:, :, :self.mhc_len_post]
        epitope_cam = cam[:, :, self.mhc_len_post:]
        
        if self.mhc_len_post != 0:
            mhc_cam = F.interpolate(mhc_cam, self.mhc_len_pre, mode="linear")
            mhc_cam = self._min_max_norm(mhc_cam, 2)
            mhc_cam = mhc_cam.view(n, self.mhc_len_pre)
        if self.epitope_len_post != 0:
            epitope_cam = F.interpolate(epitope_cam, self.epitope_len_pre, mode="linear")
            epitope_cam = self._min_max_norm(epitope_cam, 2)
            epitope_cam = epitope_cam.view(n, self.epitope_len_pre)

        return mhc_cam, epitope_cam


class GradCAMpp(CAM):
    def __init__(self, model, target_layer, len_tuple, cuda, cat_dim):
        super().__init__(model, target_layer, len_tuple, cuda, cat_dim)

    def forward(self, mhc, epitope):
        '''
        mhc: shape -> (n, encode_channel, mhc_len)
        epitope: shape -> (n, encode_channel, epitope_len)
        '''
        if self.cuda:
            self.model = self.model.cuda()
            mhc = mhc.cuda()
            epitope = epitope.cuda()

        # no sigmoid function
        x = torch.cat((self.model.modelA(mhc), self.model.modelB(epitope)), dim=self.cat_dim) #specific model architecture
        for n, m in self.model.main._modules.items(): #specific model architecture
            if isinstance(m, nn.Sigmoid):
                break
            else:
                x = m(x)

        mhc_cam, epitope_cam = self.getGradCAMpp(self.values, x)
        
        return mhc_cam.cpu().detach().numpy(), epitope_cam.cpu().detach().numpy()

    def __call__(self, mhc, epitope):
        return self.forward(mhc, epitope)

    def getGradCAMpp(self, values, x):
        '''
        values.activations: shape -> (n, C, L)
        values.gradients: shape -> (n, C, L)
        x: shape -> (n, 1)
        numerator: shape -> (n, C, L)
        denominator: shape -> (n, C, L)
        alpha: shape -> (n, C, L)
        cam: shape -> (n, 1, L)
        '''
        n, _ = x.shape

        self.model.zero_grad()
        if self.cuda:
            x.backward(torch.ones(n, 1).cuda(), retain_graph=True)
        else:
            x.backward(torch.ones(n, 1), retain_graph=True)

        # calculate alpha
        numerator = values.gradients.pow(2)
        denominator = 2 * values.gradients.pow(2)
        ag = values.activations * values.gradients.pow(3)
        denominator += ag.sum(dim=2, keepdim=True)
        denominator = torch.where(denominator != 0.0, denominator, torch.ones_like(denominator))
        alpha = numerator / (denominator + 1e-7)

        relu_grad = F.relu(x.exp().view(n,1,1) * values.gradients)
        weights = (alpha * relu_grad).sum(dim=2, keepdim=True)
        cam = (weights * values.activations).sum(dim=1, keepdim=True)
        
        # seperate cam
        mhc_cam = cam[:, :, :self.mhc_len_post]
        epitope_cam = cam[:, :, self.mhc_len_post:]
        
        if self.mhc_len_post != 0:
            mhc_cam = F.interpolate(mhc_cam, self.mhc_len_pre, mode="linear")
            mhc_cam = self._min_max_norm(mhc_cam, 2)
            mhc_cam = mhc_cam.view(n, self.mhc_len_pre)
        if self.epitope_len_post != 0:
            epitope_cam = F.interpolate(epitope_cam, self.epitope_len_pre, mode="linear")
            epitope_cam = self._min_max_norm(epitope_cam, 2)
            epitope_cam = epitope_cam.view(n, self.epitope_len_pre)

        return mhc_cam, epitope_cam


class ScoreCAM(CAM):
    def __init__(self, model, target_layer, len_tuple, cuda, cat_dim):
        super().__init__(model, target_layer, len_tuple, cuda, cat_dim)

    def forward(self, mhc, epitope):
        '''
        mhc: shape -> (n, encode_channel, mhc_len)
        epitope: shape -> (n, encode_channel, epitope_len)
        act: shape -> (n, C, L)
        weights: shape -> (n, C, 1)
        '''
        if self.cuda:
            self.model = self.model.cuda()
            mhc = mhc.cuda()
            epitope = epitope.cuda()

        with torch.no_grad():
            self.model.zero_grad()
            x = self.model(mhc, epitope)

            self.act = self.values.activations
            self.act = F.relu(self.act)
            n, C, _ = self.act.shape

            # mhc mask
            if self.mhc_len_post != 0:
                self.mhc_act = self.act[:,:,:self.mhc_len_post]
                self.mhc_act = F.interpolate(self.mhc_act, self.mhc_len_pre, mode="linear")
                self.norm_mhc_act = self._min_max_norm(self.mhc_act, 2) #(n, C, mhc_len)
                self.mhc_mask = self.norm_mhc_act.unsqueeze(dim=2) #(n, C, 1, mhc_len)
                self.mhc_mask = mhc.unsqueeze(dim=1) * self.mhc_mask #(n, C, encode_channel, mhc_len)
            else:
                self.mhc_mask = mhc.unsqueeze(dim=1).expand(-1, C, -1, -1)

            # epitope mask
            if self.epitope_len_post != 0:
                self.epitope_act = self.act[:,:,self.mhc_len_post:]
                self.epitope_act = F.interpolate(self.epitope_act, self.epitope_len_pre, mode="linear")
                self.norm_epitope_act = self._min_max_norm(self.epitope_act, 2) #(n ,C, epitope_len)
                self.epitope_mask = self.norm_epitope_act.unsqueeze(dim=2) #(n, C, 1, epitope_len)
                self.epitope_mask = epitope.unsqueeze(dim=1) * self.epitope_mask #(n, C, encode_channel, epitope_len)
            else:
                self.epitope_mask = epitope.unsqueeze(dim=1).expand(-1, C, -1, -1)

            # weights
            weights = torch.tensor([])
            if self.cuda:
                weights = weights.cuda()
            for i in range(n):
                mhc_single = self.mhc_mask[i]
                epitope_single = self.epitope_mask[i]
                probs = self.model(mhc_single, epitope_single).unsqueeze(dim=0)
                weights = torch.cat((weights, probs), dim=0)

            # mhc cam
            if self.mhc_len_post != 0:
                mhc_cam = (self.mhc_act * weights).sum(dim=1, keepdim=True) #(n, 1, mhc_len)
                # min-max normalization
                mhc_cam = self._min_max_norm(mhc_cam, 2)
                mhc_cam = mhc_cam.view(n, self.mhc_len_pre)
            else:
                mhc_cam = torch.tensor([])

            # epitope cam
            if self.epitope_len_post != 0:
                epitope_cam = (self.epitope_act * weights).sum(dim=1, keepdim=True) #(n, 1, epitope_len)
                # min-max normalization
                epitope_cam = self._min_max_norm(epitope_cam, 2)
                epitope_cam = epitope_cam.view(n, self.epitope_len_pre)
            else:
                epitope_cam = torch.tensor([])

        return mhc_cam.cpu().detach().numpy(), epitope_cam.cpu().detach().numpy()

    def __call__(self, mhc, epitope):
        return self.forward(mhc, epitope)
