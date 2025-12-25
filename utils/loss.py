import torch
import torch.nn.functional as F


wavelet_filters = {
    "haar": torch.tensor([1, 1], dtype=torch.float64) / 2**0.5,
    "db1": torch.tensor([0.48296, 0.8365, 0.22414, -0.12941], dtype=torch.float64),
    "db2": torch.tensor([0.34150635, 0.59150635, 0.15849365, -0.09150635], dtype=torch.float64),
    "db3": torch.tensor([0.2352336, 0.5705585, 0.3251825, -0.0954672, -0.0604161, 0.0249083], dtype=torch.float64),
    "db4": torch.tensor([0.162901, 0.505472, 0.446100, -0.019800, -0.132253, 0.021808, 0.023251, -0.007493], dtype=torch.float64),
}


def dwt(x, wavelet="haar"):
    if wavelet not in wavelet_filters:
        raise ValueError(f"Unsupported wavelet: {wavelet}")
    h = wavelet_filters[wavelet].to(x.device)
    lpf = h.unsqueeze(0).unsqueeze(0)
    hpf = h.flip(0).unsqueeze(0).unsqueeze(0)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    elif x.dim() == 1:
        x = x.unsqueeze(0).unsqueeze(0)
    low = F.conv1d(x, lpf, stride=2)
    high = F.conv1d(x, hpf, stride=2)
    return low, high


def idwt(low, high, wavelet="haar"):
    if wavelet not in wavelet_filters:
        raise ValueError(f"Unsupported wavelet: {wavelet}")
    h = wavelet_filters[wavelet].to(low.device)
    lpf = h.unsqueeze(0).unsqueeze(0)
    hpf = h.flip(0).unsqueeze(0).unsqueeze(0)
    low = F.conv_transpose1d(low, lpf, stride=2)
    high = F.conv_transpose1d(high, hpf, stride=2)
    return low + high


class WaveletLoss(torch.nn.Module):
    def __init__(self, wavelet="haar", alpha=0.6):
        super().__init__()
        self.wavelet = wavelet
        self.alpha = alpha

    def forward(self, input, target):
        assert input.shape == target.shape
        low_input, high_input = dwt(input, self.wavelet)
        low_target, high_target = dwt(target, self.wavelet)
        wavelet_loss = F.mse_loss(low_input, low_target) + F.mse_loss(high_input, high_target)
        mse_loss = F.mse_loss(input, target)
        return self.alpha * wavelet_loss + (1 - self.alpha) * mse_loss
