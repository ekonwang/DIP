import torch
import torch.nn.functional as F
import numpy as np

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def PerturbationLoss(output,boxes,sigma=8, use_gpu=True):
    Loss = 0.
    if boxes.shape[1] > 1:
        boxes = boxes.squeeze()
        for tempBoxes in boxes.squeeze():
            y1 = int(tempBoxes[1])
            y2 = int(tempBoxes[3])
            x1 = int(tempBoxes[2])
            x2 = int(tempBoxes[4])
            out = output[:,:,y1:y2,x1:x2]
            GaussKernel = matlab_style_gauss2D(shape=(out.shape[2],out.shape[3]),sigma=sigma)
            GaussKernel = torch.from_numpy(GaussKernel).float()
            if use_gpu: GaussKernel = GaussKernel.cuda()
            Loss += F.mse_loss(out.squeeze(),GaussKernel)
    else:
        boxes = boxes.squeeze()
        y1 = int(boxes[1])
        y2 = int(boxes[3])
        x1 = int(boxes[2])
        x2 = int(boxes[4])
        out = output[:,:,y1:y2,x1:x2]
        Gauss = matlab_style_gauss2D(shape=(out.shape[2],out.shape[3]),sigma=sigma)
        GaussKernel = torch.from_numpy(Gauss).float()
        if use_gpu: GaussKernel = GaussKernel.cuda()
        Loss += F.mse_loss(out.squeeze(),GaussKernel) 
    return Loss


def MincountLoss(output,boxes, use_gpu=True):
    ones = torch.ones(1)
    if use_gpu: ones = ones.cuda()
    Loss = 0.
    if boxes.shape[1] > 1:
        boxes = boxes.squeeze()
        for tempBoxes in boxes.squeeze():
            y1 = int(tempBoxes[1])
            y2 = int(tempBoxes[3])
            x1 = int(tempBoxes[2])
            x2 = int(tempBoxes[4])
            X = output[:,:,y1:y2,x1:x2].sum()
            if X.item() <= 1:
                Loss += F.mse_loss(X,ones)
    else:
        boxes = boxes.squeeze()
        y1 = int(boxes[1])
        y2 = int(boxes[3])
        x1 = int(boxes[2])
        x2 = int(boxes[4])
        X = output[:,:,y1:y2,x1:x2].sum()
        if X.item() <= 1:
            Loss += F.mse_loss(X,ones)  
    return Loss