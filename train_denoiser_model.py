import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ NAFNet parts
class LayerNorm2d(nn.Module):
    '''Channel-wise LayerNorm over NCHW. Implemented via a permute so it hits the fused
    F.layer_norm kernel; with channels_last tensors the permute is a free view.'''

    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.normalized_shape = (c,)
        self.eps = eps

    def forward(self, x):
        y = F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape,
                         self.weight, self.bias, self.eps)
        return y.permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2, drop_out_rate=0.0):
        super().__init__()
        dw = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.conv3 = nn.Conv2d(dw // 2, c, 1)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        self.sg = SimpleGate()

        ffn = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.drop1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.drop2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x = self.conv1(self.norm1(inp))
        x = self.sg(self.conv2(x))
        x = x * self.sca(x)
        x = self.drop1(self.conv3(x))
        y = inp + x * self.beta

        x = self.sg(self.conv4(self.norm2(y)))
        x = self.drop2(self.conv5(x))
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, width=32, enc_blks=(2, 2, 4, 6), middle_blk_num=8,
                 dec_blks=(2, 2, 2, 2), in_ch=3, zero_init_out=True):
        super().__init__()
        self.intro = nn.Conv2d(in_ch, width, 3, padding=1)
        self.ending = nn.Conv2d(width, in_ch, 3, padding=1)

        self.encoders, self.downs = nn.ModuleList(), nn.ModuleList()
        self.decoders, self.ups = nn.ModuleList(), nn.ModuleList()

        c = width
        for n in enc_blks:
            self.encoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, c * 2, 2, 2))
            c *= 2
        self.middle = nn.Sequential(*[NAFBlock(c) for _ in range(middle_blk_num)])
        for n in dec_blks:
            self.ups.append(nn.Sequential(nn.Conv2d(c, c * 2, 1, bias=False),
                                          nn.PixelShuffle(2)))
            c //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))

        self.padder = 2 ** len(enc_blks)
        if zero_init_out:
            nn.init.zeros_(self.ending.weight)
            nn.init.zeros_(self.ending.bias)

    def _pad(self, x):
        _, _, h, w = x.shape
        ph = (self.padder - h % self.padder) % self.padder
        pw = (self.padder - w % self.padder) % self.padder
        return F.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x

    def forward(self, inp):
        h, w = inp.shape[-2:]
        x0 = self._pad(inp)
        x = self.intro(x0)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x); skips.append(x); x = down(x)
        x = self.middle(x)
        for dec, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = up(x) + skip
            x = dec(x)
        x = self.ending(x) + x0
        return x[..., :h, :w]


# ------------------------------------------------------------------ v3 legacy net
class _ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class _Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1),
                                  nn.SiLU(inplace=True), _ResBlock(cout))
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        f = self.conv(x)
        return f, self.pool(f)


class _Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 2, stride=2)
        self.conv = nn.Sequential(nn.Conv2d(cout + cskip, cout, 3, padding=1),
                                  nn.SiLU(inplace=True), _ResBlock(cout))

    def forward(self, x, skip):
        return self.conv(torch.cat([self.up(x), skip], dim=1))


class UNetV3(nn.Module):
    '''The v3 residual U-Net, kept so old checkpoints still load.'''

    def __init__(self, base=48):
        super().__init__()
        b = base
        self.e1, self.e2 = _Down(3, b), _Down(b, b * 2)
        self.e3, self.e4 = _Down(b * 2, b * 4), _Down(b * 4, b * 8)
        self.bridge = nn.Sequential(nn.Conv2d(b * 8, b * 16, 3, padding=1),
                                    nn.SiLU(inplace=True), _ResBlock(b * 16))
        self.d4 = _Up(b * 16, b * 8, b * 8)
        self.d3 = _Up(b * 8, b * 4, b * 4)
        self.d2 = _Up(b * 4, b * 2, b * 2)
        self.d1 = _Up(b * 2, b, b)
        self.final = nn.Conv2d(b, 3, 1)
        nn.init.zeros_(self.final.weight); nn.init.zeros_(self.final.bias)

    def forward(self, x):
        s1, h = self.e1(x); s2, h = self.e2(h); s3, h = self.e3(h); s4, h = self.e4(h)
        h = self.bridge(h)
        h = self.d4(h, s4); h = self.d3(h, s3); h = self.d2(h, s2); h = self.d1(h, s1)
        return x + self.final(h)


# ------------------------------------------------------------------ public wrapper
class FaceDenoiseNet(nn.Module):
    '''arch='nafnet' (v4, default) or 'unet' (v3 legacy).

    Output is clamped to [0,1] in eval only - clamping during training zeroes the
    gradient wherever the residual overshoots, which slows convergence for no gain.
    '''

    def __init__(self, base=32, arch="nafnet", enc=(2, 2, 4, 6), mid=8, dec=(2, 2, 2, 2)):
        super().__init__()
        self.arch = arch
        self.base = base
        self.cfg = dict(base=base, arch=arch, enc=tuple(enc), mid=mid, dec=tuple(dec))
        if arch == "nafnet":
            self.net = NAFNet(width=base, enc_blks=tuple(enc),
                              middle_blk_num=mid, dec_blks=tuple(dec))
        elif arch == "unet":
            self.net = UNetV3(base=base)
        else:
            raise ValueError(f"unknown arch {arch}")

    def forward(self, x):
        y = self.net(x)
        return y if self.training else y.clamp(0.0, 1.0)


def build_from_ckpt_meta(ck):
    return FaceDenoiseNet(
        base=ck.get("base_width", 32),
        arch=ck.get("arch", "unet"),
        enc=tuple(ck.get("enc", (2, 2, 4, 6))),
        mid=ck.get("mid", 8),
        dec=tuple(ck.get("dec", (2, 2, 2, 2))))


def load_denoiser(path, device="cpu", prefer_ema=True):
    '''Loads v4 or v3 checkpoints. Returns an eval-mode FaceDenoiseNet.'''
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = None
    if prefer_ema and "ema_state_dict" in ck:
        sd = ck["ema_state_dict"]
    elif "model_state_dict" in ck:
        sd = ck["model_state_dict"]
    elif "state_dict" in ck:
        sd = ck["state_dict"]
    else:
        sd = ck
    net = build_from_ckpt_meta(ck if isinstance(ck, dict) else {})
    try:
        net.load_state_dict(sd)
    except Exception:
        # v3 checkpoints were saved without the wrapper prefix
        net.load_state_dict({f"net.{k}": v for k, v in sd.items()})
    net.to(device).eval()
    net.meta = {k: v for k, v in ck.items() if not k.endswith("state_dict")} \
        if isinstance(ck, dict) else {}
    return net


@torch.no_grad()
def restore_image(net, bgr, device="cuda", amp=True, tile=0, overlap=32):
    '''bgr: HxWx3 uint8 (OpenCV order). Returns HxWx3 uint8. Any size; padding is internal.
    Set tile=512 for frames too big to fit in VRAM.'''
    import numpy as np
    x = torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1])).to(device)
    x = x.permute(2, 0, 1).float().div_(255).unsqueeze(0)
    x = x.contiguous(memory_format=torch.channels_last)
    dt = torch.bfloat16 if (amp and torch.cuda.is_available()
                            and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    ctx = torch.autocast("cuda", dtype=dt, enabled=amp and str(device).startswith("cuda"))

    def run(t):
        with ctx:
            return net(t).float()

    if tile and max(x.shape[-2:]) > tile:
        _, _, H, W = x.shape
        out = torch.zeros_like(x)
        wsum = torch.zeros(1, 1, H, W, device=x.device)
        step = tile - overlap
        for top in range(0, H, step):
            for left in range(0, W, step):
                b, r = min(top + tile, H), min(left + tile, W)
                t0, l0 = max(0, b - tile), max(0, r - tile)
                out[..., t0:b, l0:r] += run(x[..., t0:b, l0:r])
                wsum[..., t0:b, l0:r] += 1
        y = out / wsum.clamp_min(1)
    else:
        y = run(x)

    y = y.clamp(0, 1)[0].permute(1, 2, 0).mul_(255).round_().byte().cpu().numpy()
    return y[:, :, ::-1].copy()
