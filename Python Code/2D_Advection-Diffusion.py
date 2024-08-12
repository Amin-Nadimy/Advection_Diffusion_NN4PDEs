#  Copyright (C) 2024
#  
#  Amin Nadimy, Boyang Chen, Christopher Pain
#  Applied Modelling and Computation Group
#  Department of Earth Science and Engineering
#  Imperial College London
#  ++++++++++++++++++++++++++++++++++++++++
#  amin.nadimy19@imperial.ac.uk
#  
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation,
#  version 3.0 of the License.
#
#  This library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  Lesser General Public License for more details.

# import required libraries
import sys
assert sys.version_info >= (3, 5)
import torch
import os
import matplotlib.pyplot as plt
import time 
from torch import nn

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    num_gpu_devices = torch.cuda.device_count()
    device_names = [torch.cuda.get_device_name(i) for i in range(num_gpu_devices)]

    print(f"Number of available GPU devices: {num_gpu_devices}")
    device = []
    for i, device_name in enumerate(device_names):
        device.append(torch.device(f"cuda:{i}"))
        print(f"GPU {i}: {device_name}, {device[i]}")
        
else:
    device = 'cpu'
    print("No GPU devices available. Using CPU.")
is_gpu = torch.cuda.is_available()


# # # ################################### # # #
# # # ######    Model Parameters    ##### # # #
# # # ################################### # # #
nx = 128
ny = 128 
u_x = 5.
u_y = 5.

ntime = 2000                        # number of timesteps
ctime = 0                           # Computatioanl time
k = 0.5                             # diffusion coeficient

dx = 1
dy = 1
cfl = 0.05
dt = cfl*dx/u_x


# # # ################################### # # #
# # # ######      CNN Filters       ##### # # #
# # # ################################### # # #
diff  = torch.tensor([[1/3,  1/3 , 1/3],
                      [1/3, -8/3 , 1/3],
                      [1/3,  1/3 , 1/3]]).unsqueeze(0).unsqueeze(0)

xadv = torch.tensor([[1/12, 0.0, -1/12],
                     [1/3 , 0.0, -1/3] ,
                     [1/12, 0.0, -1/12]]).unsqueeze(0).unsqueeze(0)

yadv = torch.tensor([[-1/12, -1/3, -1/12],
                     [ 0.0  , 0.0 , 0.0]  ,
                     [ 1/12 , 1/3 , 1/12]]).unsqueeze(0).unsqueeze(0)
ml = 0.25
# Bias :: mean b=0 in y=ax+b
bias_initializer = torch.tensor([0.0])


# # # ################################### # # #
# # # ######   Initial Condition    ##### # # #
# # # ################################### # # #
input_shape = (1,1,ny,nx) # 1st one  is for number of input images, hight, width , and 1==number of color channel
values = torch.zeros(input_shape, device=device)
for i in range(40):
    for j in range(40):
        values[0,0,i+43,j+43] = 1 


# # # ################################### # # #
# # # ######         Model          ##### # # #
# # # ################################### # # #
class adv_diff(nn.Module):
    def __init__(self):
        super(adv_diff, self).__init__()
        self.diff = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.xadv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.yadv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        
        self.diff.weight.data = diff  # Replace with your initializer
        self.xadv.weight.data = xadv  # Replace with your initializer
        self.yadv.weight.data = yadv  # Replace with your initializer
        
        self.diff.bias.data = bias_initializer  # Replace with your initializer
        self.xadv.bias.data = bias_initializer  # Replace with your initializer
        self.yadv.bias.data = bias_initializer  # Replace with your initializer

    def forward(self, values):
        values =  values + (k*self.diff(values)/dx**2+self.xadv(values)/dx+self.yadv(values)/dx)/ml*dt
        return values

model = adv_diff().to(device)


# # # ################################### # # #
# # # ######         Main           ##### # # #
# # # ################################### # # #
istep=0
start = time.time()


with torch.no_grad():
    for itime in range(1,ntime+1): 
        istep +=1
        ctime = ctime + dt 
        values = model(values)

end = time.time()
print('itime, istep:', itime, istep) 
print(f'time:{(end-start):.0f}s, {ctime:.1f}s')