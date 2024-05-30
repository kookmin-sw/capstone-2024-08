import torch
from models import SynthesizerTrn

net_g = SynthesizerTrn(
    n_vocab=256,
    spec_channels=80, # <--- vits2 parameter (changed from 513 to 80)
    segment_size=8192,
    inter_channels=192,
    hidden_channels=192,
    filter_channels=768,
    n_heads=2,
    n_layers=6,
    kernel_size=3,
    p_dropout=0.1,
    resblock="1", 
    resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    upsample_rates=[8, 8, 2, 2],
    upsample_initial_channel=512,
    upsample_kernel_sizes=[16, 16, 4, 4],
    n_speakers=0,
    # speaker cond layer input size
    # self.spk_emb_linear = nn.Linear(self.gin_channels, self.hidden_channels)
    gin_channels=0,
    use_sdp=True, 
    use_transformer_flows=True, # <--- vits2 parameter
    # (choose from "pre_conv", "fft", "mono_layer_inter_residual", "mono_layer_post_residual")
    transformer_flow_type="fft", # <--- vits2 parameter 
    use_spk_conditioned_encoder=True, # <--- vits2 parameter
    use_noise_scaled_mas=True, # <--- vits2 parameter
    use_duration_discriminator=True, # <--- vits2 parameter
)

x = torch.LongTensor([[1, 2, 3],[4, 5, 6]]) # token ids
x_lengths = torch.LongTensor([3, 2]) # token lengths
y = torch.randn(2, 80, 100) # mel spectrograms
y_lengths = torch.Tensor([100, 80]) # mel spectrogram lengths

net_g(
    x=x,
    x_lengths=x_lengths,
    y=y,
    y_lengths=y_lengths,
)

# calculate loss and backpropagate