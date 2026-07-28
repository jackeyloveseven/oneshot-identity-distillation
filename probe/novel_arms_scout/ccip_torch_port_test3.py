import time
import numpy as np
import torch
from onnx2torch import convert

torch_model = convert("ccip_feat_simplified_bs4.onnx")
torch_model.eval()
n_params = sum(p.numel() for p in torch_model.parameters())
print(f"n_params={n_params:,}")

from imgutils.metrics.ccip import _preprocess_image, _open_feat_model
from PIL import Image
import glob
cands = sorted(glob.glob('/mnt/local/smoke_anime/anime_260623/img/*_ref.png'))[:4]
print("test images:", cands)
data_b = np.stack([_preprocess_image(Image.open(p).convert('RGB'), size=384) for p in cands]).astype(np.float32)
print("input batch shape:", data_b.shape)

sess = _open_feat_model('ccip-caformer-24-randaug-pruned')
onnx_out, = sess.run(['output'], {'input': data_b})
print("onnxruntime bs=4 output:", onnx_out.shape)

torch_in = torch.from_numpy(data_b)
with torch.no_grad():
    torch_out = torch_model(torch_in)
torch_out_t = torch_out[0] if isinstance(torch_out, (tuple, list)) else torch_out
diff = np.abs(onnx_out - torch_out_t.numpy())
print(f"CPU bs=4 numeric check -- max abs diff: {diff.max():.6e} mean abs diff: {diff.mean():.6e}")

# --- differentiability check ---
torch_in_g = torch.from_numpy(data_b).clone().requires_grad_(True)
out = torch_model(torch_in_g)
out_t = out[0] if isinstance(out, (tuple, list)) else out
loss = out_t.pow(2).mean()
loss.backward()
print("grad exists:", torch_in_g.grad is not None, "grad finite:", torch.isfinite(torch_in_g.grad).all().item() if torch_in_g.grad is not None else None)

# --- CPU timing (fwd only, fwd+bwd) ---
N = 5
with torch.no_grad():
    for _ in range(2): torch_model(torch_in)  # warmup
    t0 = time.time()
    for _ in range(N): torch_model(torch_in)
    cpu_fwd = (time.time()-t0)/N
print(f"CPU bs=4 forward (no_grad): {cpu_fwd*1000:.1f} ms/iter")

t0 = time.time()
for _ in range(N):
    x = torch_in.clone().requires_grad_(True)
    o = torch_model(x)
    ot = o[0] if isinstance(o,(tuple,list)) else o
    ot.pow(2).mean().backward()
cpu_fwdbwd = (time.time()-t0)/N
print(f"CPU bs=4 forward+backward: {cpu_fwdbwd*1000:.1f} ms/iter")

# --- GPU timing if available ---
if torch.cuda.is_available():
    free0, total0 = torch.cuda.mem_get_info()
    print(f"GPU free/total before load: {free0/1e9:.2f}/{total0/1e9:.2f} GB")
    torch.cuda.reset_peak_memory_stats()
    gm = torch_model.to("cuda")
    x_gpu = torch_in.to("cuda")
    with torch.no_grad():
        for _ in range(3): gm(x_gpu)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(N): gm(x_gpu)
    torch.cuda.synchronize()
    gpu_fwd = (time.time()-t0)/N
    peak_fwd = torch.cuda.max_memory_allocated()/1e9
    print(f"GPU bs=4 forward (no_grad): {gpu_fwd*1000:.1f} ms/iter, peak_alloc_extra={peak_fwd:.3f} GB")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(N):
        xg = x_gpu.clone().requires_grad_(True)
        o = gm(xg)
        ot = o[0] if isinstance(o,(tuple,list)) else o
        ot.pow(2).mean().backward()
    torch.cuda.synchronize()
    gpu_fwdbwd = (time.time()-t0)/N
    peak_fwdbwd = torch.cuda.max_memory_allocated()/1e9
    print(f"GPU bs=4 forward+backward: {gpu_fwdbwd*1000:.1f} ms/iter, peak_alloc_extra={peak_fwdbwd:.3f} GB")
    free1, total1 = torch.cuda.mem_get_info()
    print(f"GPU free/total after: {free1/1e9:.2f}/{total1/1e9:.2f} GB")
else:
    print("no CUDA available in this process")
