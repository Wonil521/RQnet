import torch
import torch.nn as nn
import torch.nn.functional as F

def inv_stn(x, theta, device):
    h=x.size(2)
    w=x.size(3)
    dummy_array = torch.tensor([[0.,0.,1.]]).expand(theta.size(0), theta.size(1),1,3).cuda(device)
    inv_theta_matrix = torch.zeros(theta.size(0), theta.size(1), 3, 3)

    theta_tmp = torch.cat((theta, dummy_array),2)
    inv_theta_matrix = torch.inverse(theta_tmp.view(-1,3,3))
    inv_theta = inv_theta_matrix[:,0:2,:].view(-1,2,3)

    x_tmp = x.reshape(-1,1,h,w)
    grid = F.affine_grid(inv_theta, x_tmp.size())
    S_estimate_inverse_aligned = F.grid_sample(x_tmp, grid.cuda(device), mode='bilinear', padding_mode='reflection')

    return S_estimate_inverse_aligned, inv_theta


class Net_x(nn.Module):
    def __init__(self, num_bvalue):
        super().__init__()
        self.num_bvalue = num_bvalue
        self.layer1 = nn.Conv2d(num_bvalue, 30, kernel_size=1,stride=1) #input_channel = num_bvalues
        self.layer2 = nn.Conv2d(30, 30, kernel_size=1,stride=1)
        self.layer3 = nn.Conv2d(30, 30, kernel_size=1,stride=1)
        self.layer4 = nn.Conv2d(30, 10, kernel_size=1,stride=1)
        self.layer5 = nn.Conv2d(10, 4, kernel_size=1,stride=1)
        self.relu = nn.ReLU()
        self.ELU  = nn.ELU()
        self.sigmoid = nn.Sigmoid()

        # =============================STN=======================================
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(7840, 32), #128 x 128 input
            # nn.Linear(7290, 32),  # 122 x 122 input
            nn.ReLU(True),
            # nn.Linear(32, 3 * 2 * (num_bvalue-1))
            nn.Linear(32, 3 * 2),
            # nn.Sigmoid()
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1., 0., 0., 0., 1., 0.], dtype=torch.float))
        # self.fc_loc[2].bias.data.copy_(torch.tensor([0.,0.,0.,0.,0.,0.], dtype=torch.float))

        #Quantification Network for f
        self.quant_net_f = nn.Sequential(
            nn.Conv2d(num_bvalue, 30, kernel_size=1, stride=1),
            nn.ELU(True),
            nn.Conv2d(30, 30, kernel_size=3, stride=1, padding=(1,1)),
            nn.ELU(True),
            nn.Conv2d(30, 10, kernel_size=1, stride=1),
            nn.ELU(True),
            nn.Conv2d(10, 1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

        #Quantification Network
        self.quant_net = nn.Sequential(
            nn.Conv2d(num_bvalue, 30, kernel_size=1, stride=1),
            nn.ELU(True),
            nn.Conv2d(30, 30, kernel_size=3, stride=1, padding=(1,1)),
            nn.ELU(True),
            nn.Conv2d(30, 10, kernel_size=1, stride=1),
            nn.ELU(True),
            nn.Conv2d(10, 4, kernel_size=1, stride=1),
            nn.Sigmoid()
        )


    def stn(self, x):
        h=x.size(2)
        w=x.size(3)
        for i in range(self.num_bvalue-1):
            xi = torch.cat((x[:,0,:,:].view(-1,1,h,w), x[:,i+1,:,:].view(-1,1,h,w)),1) # (B, C, H, W)
            xs = self.localization(xi)

            xs = xs.view(-1, 7840) # 128 x 128 input
            # xs = xs.view(-1, 7290)  # 122 x 122 input
            theta = self.fc_loc(xs) # (batch, 6)
            theta = theta.view(-1,2,3) # (batch, 2, 3)

            # theta_max = torch.tensor([1.05, 0.5, 0.05, 0.5, 1.05, 0.05]).reshape([1,2,3]).cuda(torch.device('cuda:4'))
            # theta_min = torch.tensor([0.95, -0.5, -0.05, -0.5, 0.95, -0.05]).reshape([1,2,3]).cuda(torch.device('cuda:4'))
            # theta = (theta_max-theta_min)*theta+theta_min

            if i==0:
                theta_a = theta.view(-1,1,2,3)
            else:
                theta_a = torch.cat((theta_a,theta.view(-1,1,2,3)),1)

        grid = F.affine_grid(theta_a.view(-1,2,3), x[:,1:,:,:].reshape(-1,1,h,w).size()) # (N,C,H,W)
        motion_corrected_S_b = F.grid_sample(x[:,1:,:,:].reshape(-1,1,h,w), grid, mode='bilinear', padding_mode='reflection') # (N,C,H,W)

        motion_corrected_S_b = motion_corrected_S_b.view(-1,self.num_bvalue-1,h,w)
        motion_corrected_S = torch.cat((x[:,0,:,:].view(-1,1,h,w), motion_corrected_S_b),1)

        theta_a = theta_a.view(-1, (self.num_bvalue-1), 2, 3)
        return motion_corrected_S, theta_a

    def forward(self, x, S0mask):
        h=x.size(2)
        w=x.size(3)

        #STN
        motion_corrected_S, theta = self.stn(x)
        motion_corrected_S_masked = motion_corrected_S*S0mask

        #x1 = registered input
        x1 = motion_corrected_S_masked
        # Out_x = self.quant_net(torch.cat((x1[:,0,:,:].view(-1,1,h,w), x1[:,1:,:,:]), 1)) #x
        Out_x = self.quant_net(x1) #x

        #Estimate IVIM parameters from transformed DWIs
        f = self.quant_net_f(x1)

        return Out_x, f, theta, motion_corrected_S, motion_corrected_S_masked