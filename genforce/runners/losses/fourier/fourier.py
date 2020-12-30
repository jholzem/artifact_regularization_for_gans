import torch

# utility function to convert RGB into gray-scale images with COLOR CHANNELS IN LAST DIMENSION
def rgb2gray(images):
    return 0.299*images[:, :, :, 0] + 0.587*images[:, :, :, 1] + 0.114*images[:, :, :, 2]


# call it as 'loss = ... + torch.mean(fourier_dissimilarity(fake_images, real_images, metric))'
# metric can be one of the following: '1' (1-norm of difference),
#                                     '2' (frobenius norm of difference),
#                                     'cos' (cosine dissimilarity)
def fourier_dissimilarity(fake_images, real_images, metric, thres=1):
    fake_images = fake_images.permute(0, 2, 3, 1)
    real_images = real_images.permute(0, 2, 3, 1)
    fake_ft = torch.norm(torch.rfft(rgb2gray(fake_images), signal_ndim=2), dim=3)
    real_ft = torch.norm(torch.rfft(rgb2gray(real_images), signal_ndim=2), dim=3)
    if metric == '1':
        return torch.norm((fake_ft[:,thres:-thres,thres:]-real_ft[:,thres:-thres,thres:]).cpu(), p=1, dim=(1, 2))*1e-8
    elif metric == '2':
        return torch.norm((fake_ft[:,thres:-thres,thres:]-real_ft[:,thres:-thres,thres:]).cpu(), p='fro', dim=(1, 2))*2e-4
    elif metric == 'cos':
        vec_fake_ft = torch.flatten(fake_ft[:,thres:-thres,thres:], start_dim=1).unsqueeze(dim=1)
        vec_real_ft = torch.flatten(real_ft[:,thres:-thres,thres:], start_dim=1).unsqueeze(dim=2)
        return 1 - torch.bmm(vec_fake_ft, vec_real_ft).squeeze() / (torch.norm(vec_fake_ft, dim=2) * torch.norm(vec_real_ft, dim=1)).squeeze()
    else:
        return 0
