"""BEVSegNet: camera-only BEV semantic segmentation at a fine output cell (e.g. 0.1 m).

image      a frozen foundation ViT (DINOv2 / DINOv3 via Hugging Face; its last 4 layers through a trained adapter)
           and a trained ResNet18 stem: features at stride 8 (ViT + ResNet layer2) and stride 2 (ResNet conv1)
lift       every BEV cell (context cell, e.g. 0.4 m) is sampled from every camera at several heights above the
           ground (SimpleBEV, Harley et al. 2023): a point on the road and an object standing on it project to
           different pixels, so the stacked heights let the network tell ground texture from things above it -
           sampling the ground plane only smears every object along its camera ray. Cameras are averaged with a
           weight that falls with the depth (the closest view has the sharpest pixels)
BEV        U-Net over the context grid (3 scales, dilated residual blocks): straight, connected shapes need context
           beyond the receptive field of the lifted samples
detail     every output cell samples the stride-2 features at the ground plane, weighted by depth the same way,
           together with log(ground footprint of a pixel); a small decoder adds a residual to the upsampled context
           logits - near the cameras it sharpens edges and thin lines, far away (footprint >> cell) it learns to
           trust the context
temporal   (history > 1) earlier frames are lifted into the current ego frame through the ego motion, mixed like the
           current frame and fused oldest -> current by a convolutional GRU: the gates learn per cell which frame to
           trust (an earlier view is farther from cells ahead, closer to cells behind, and sees what a vehicle hides
           now). Only past frames: the model stays usable online. The image encoder runs without gradient on them
known      output channel 0 is trained as "cell unlabelled": the model can abstain (metrics.predict)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from hdlabel.bevnet.geometry import grid, lift, project

from .data import NC

MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def gn(c):
    return nn.GroupNorm(8, c)


class Res(nn.Module):
    def __init__(self, c, dil=1):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(c, c, 3, padding=dil, dilation=dil), gn(c), nn.ReLU(),
                               nn.Conv2d(c, c, 3, padding=1), gn(c))  # fmt: skip

    def forward(self, x):
        return F.relu(x + self.b(x))


def conv(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, stride=s, padding=1), gn(o), nn.ReLU())


class ImageEncoder(nn.Module):
    """images [N, 3, H, W] (normalised) -> stride 8 [N, 128, H/8, W/8] and stride 2 [N, 64, H/2, W/2] features."""

    def __init__(self, vit: str | None):
        super().__init__()
        try:
            r = torchvision.models.resnet18(weights="DEFAULT")
        except Exception:  # offline
            r = torchvision.models.resnet18(weights=None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)  # stride 2, 64
        self.body = nn.Sequential(r.maxpool, r.layer1, r.layer2)  # stride 8, 128
        self.vit = None
        if vit:
            from transformers import AutoModel

            self.vit = AutoModel.from_pretrained(vit).eval().requires_grad_(False)
            cfg = self.vit.config
            self.patch = cfg.patch_size
            self.skip = 1 + int(getattr(cfg, "num_register_tokens", 0) or 0)  # CLS + register tokens
            dv = cfg.hidden_size
            self.adapter = nn.Sequential(
                nn.Conv2d(4 * dv, 256, 1), gn(256), nn.ReLU(), nn.Conv2d(256, 128, 3, padding=1)
            )

    def train(self, mode=True):
        super().train(mode)
        if self.vit is not None:
            self.vit.eval()  # frozen
        return self

    def forward(self, u):
        s2 = self.stem(u)
        s8 = self.body(s2)
        if self.vit is not None:
            h, w = u.shape[-2] // self.patch * self.patch, u.shape[-1] // self.patch * self.patch
            v = F.interpolate(u, size=(h, w), mode="bilinear", align_corners=False)
            with torch.no_grad():
                hs = self.vit(pixel_values=v, output_hidden_states=True).hidden_states[-4:]
            t = torch.cat([x[:, self.skip :] for x in hs], -1).transpose(1, 2)
            t = t.reshape(len(u), -1, h // self.patch, w // self.patch)
            t = self.adapter(t.to(self.adapter[0].weight.dtype))
            s8 = s8 + F.interpolate(t, size=s8.shape[-2:], mode="bilinear", align_corners=False)
        return s8, s2


class ConvGRU(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.zr = nn.Conv2d(2 * c, 2 * c, 3, padding=1)
        self.h = nn.Conv2d(2 * c, c, 3, padding=1)

    def forward(self, x, h):
        z, r = torch.sigmoid(self.zr(torch.cat([x, h], 1))).chunk(2, 1)
        return (1 - z) * h + z * torch.tanh(self.h(torch.cat([x, r * h], 1)))


class BEVUNet(nn.Module):
    def __init__(self, cin, c=128):
        super().__init__()
        self.inp = nn.Sequential(conv(cin, c), Res(c))
        self.d1 = nn.Sequential(conv(c, 192, 2), Res(192), Res(192, 2))
        self.d2 = nn.Sequential(conv(192, 256, 2), Res(256), Res(256, 2), Res(256, 4))
        self.u1 = nn.Sequential(conv(256 + 192, 192), Res(192))
        self.u0 = nn.Sequential(conv(192 + c, c), Res(c), Res(c, 2))

    def forward(self, x):
        up = lambda a, b: F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False)
        e0 = self.inp(x)
        e1 = self.d1(e0)
        e2 = self.d2(e1)
        d1 = self.u1(torch.cat([up(e2, e1), e1], 1))
        return self.u0(torch.cat([up(d1, e0), e0], 1))


class BEVSegNet(nn.Module):
    def __init__(self, bev: dict, n_cams: int, image_size=(704, 448), cell: float = 0.4, out_res: float | None = 0.1,
                 heights=(-0.5, 0.0, 0.5, 1.0, 2.0), vit: str | None = "facebook/dinov2-base", jitter: float = 0.0,
                 history: int = 1):  # fmt: skip
        super().__init__()
        self.config = dict(bev=dict(bev), n_cams=n_cams, image_size=list(image_size), cell=cell, out_res=out_res,
                           heights=list(heights), vit=vit, jitter=jitter, history=history)  # fmt: skip
        self.size, self.heights, self.jitter, self.history = tuple(image_size), list(heights), jitter, history
        self.enc = ImageEncoder(vit)
        self.hmix = nn.Sequential(nn.Conv2d(128 * len(self.heights) + 2, 128, 1), gn(128), nn.ReLU())
        self.gru = ConvGRU(128) if history > 1 else None
        self.bev = BEVUNet(128)
        self.ctx_head = nn.Conv2d(128, NC, 1)
        self.register_buffer("Pc", grid(bev, cell), persistent=False)
        self.fc = int(round(cell / bev["res"]))  # label cells per context cell
        self.f = self.fc
        self.out_res = out_res
        if out_res:
            self.register_buffer("Pd", grid(bev, out_res), persistent=False)
            self.f = int(round(out_res / bev["res"]))
            self.dctx = nn.Conv2d(128, 32, 1)
            self.detail = nn.Sequential(conv(64 + 32 + NC + 1, 64), Res(64), Res(64, 2), nn.Conv2d(64, NC, 1))
        self.vis_depth = None
        self.ctx_logits = None

    # -- images ------------------------------------------------------------------------------------------------------
    def normalise(self, img):
        """uint8 [B, C, 3, H, W] -> normalised float, with brightness / contrast jitter per camera when training."""
        u = img.float() / 255
        if self.training and self.jitter:
            B, C = u.shape[:2]
            gain = 1 + (torch.rand(B, C, 1, 1, 1, device=u.device) - 0.5) * 2 * self.jitter
            bias = (torch.rand(B, C, 1, 1, 1, device=u.device) - 0.5) * self.jitter
            u = (u * gain + bias).clamp(0, 1)
        return (u - MEAN.to(u.device)) / STD.to(u.device)

    @staticmethod
    def sample(feat, g, ok, wt):
        """feat [B, C, F, h, w] sampled at g [B, C, N, 2], weighted mean over the cameras (ok [B, C, N] in view,
        wt [B, C, N] weight) -> [B, F, N]."""
        acc = wsum = 0
        for c in range(feat.shape[1]):  # camera by camera: [B, F, N] at a time
            s = F.grid_sample(feat[:, c], g[:, c, None].to(feat.dtype), align_corners=False)[:, :, 0]
            w = (ok[:, c] * wt[:, c]).to(s.dtype)[:, None]
            acc = acc + s * w
            wsum = wsum + w
        return acc / wsum.clamp(min=1e-6)

    # -- forward -----------------------------------------------------------------------------------------------------
    def forward(self, x):
        dev = self.Pc.device
        frames = x["images"][:, : self.history].to(dev, non_blocking=True)  # [B, T, C, 3, H, W], current first
        img = frames[:, 0]
        B, C = img.shape[:2]
        Tcf, Tbc, K, cm = (x[k].to(dev) for k in ("T_cur_frame", "T_base_cam", "K", "cam_mask"))
        Tcf = Tcf[:, : self.history]
        A = x["A"].to(dev) if "A" in x else torch.eye(4, device=dev).expand(B, 4, 4)
        f8, f2 = self.enc(self.normalise(img).flatten(0, 1))
        f8, f2 = f8.view(B, C, *f8.shape[1:]), f2.view(B, C, *f2.shape[1:])

        X, Y = self.Pc.shape[:2]
        with torch.autocast("cuda", enabled=False):  # geometry in float32
            G, Z, OK = project(lift(self.Pc, self.heights, A), Tcf, Tbc, K, cm, self.size)  # [B, C, T, H*X*Y, ...]
        i0 = self.heights.index(0.0) if 0.0 in self.heights else 0
        ground = slice(i0 * X * Y, (i0 + 1) * X * Y)

        def frame(f8, t):
            """BEV features of frame t (lifted at every height, with its view geometry) -> [B, 128, X, Y]; also the
            depth of the closest camera that sees each ground cell."""
            g, z, ok = G[:, :, t], Z[:, :, t], OK[:, :, t]
            lifted = self.sample(f8, g, ok, 1 / z.clamp(min=1)).view(B, -1, len(self.heights), X, Y).flatten(1, 2)
            z0, ok0 = z[:, :, ground], ok[:, :, ground]
            vis = torch.where(ok0, z0, torch.inf).amin(1).view(B, X, Y)
            geo = torch.stack([ok0.float().mean(1), torch.log(vis.clamp(1, 200)).flatten(1) / 5], 1)
            return self.hmix(torch.cat([lifted, geo.view(B, 2, X, Y).to(lifted.dtype)], 1)), vis

        m, self.vis_depth = frame(f8, 0)
        if self.gru is not None:
            h = torch.zeros_like(m)
            for t in range(frames.shape[1] - 1, 0, -1):  # oldest first
                with torch.no_grad():
                    ft = self.enc(self.normalise(frames[:, t]).flatten(0, 1))[0]
                h = self.gru(frame(ft.view(B, C, *ft.shape[1:]), t)[0], h)
            m = self.gru(m, h)
        h = self.bev(m)
        ctx = self.ctx_head(h)
        if not self.out_res:
            self.ctx_logits = None
            return ctx
        self.ctx_logits = ctx

        Xd, Yd = self.Pd.shape[:2]
        with torch.autocast("cuda", enabled=False):
            gd, zd, okd = project(lift(self.Pd, [0.0], A), Tcf, Tbc, K, cm, self.size)
        gd, zd, okd = gd[:, :, 0], zd[:, :, 0], okd[:, :, 0]
        det = self.sample(f2, gd, okd, 1 / zd.clamp(min=1) ** 2).view(B, -1, Xd, Yd)
        fx = K[:, :, 0, 0][:, :, None]
        foot = torch.where(okd, 2 * zd / fx, torch.inf).amin(1).clamp(max=10).view(B, 1, Xd, Yd)  # m per stride-2 px
        up = F.interpolate(ctx, size=(Xd, Yd), mode="bilinear", align_corners=False)
        hc = F.interpolate(self.dctx(h), size=(Xd, Yd), mode="bilinear", align_corners=False)
        feats = torch.cat([det.to(up.dtype), hc, up, torch.log(foot.clamp(min=0.01)).to(up.dtype)], 1)
        return up + self.detail(feats)
