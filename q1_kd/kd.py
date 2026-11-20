"""
Knowledge distillation framework.

Total student loss
    L = (1 - alpha) * CE(z_s, y)                                   hard-label term
      + alpha * T^2 * KL( softmax(z_t / T) || softmax(z_s / T) )   soft-target term (Hinton et al., 2015)
      + beta  * sum_i  L_feat(f_s^i, f_t^i)                        feature term

* Temperature T > 1 softens both distributions so the student also learns the
  teacher's "dark knowledge" (relative probabilities of wrong classes). The T^2
  factor keeps the soft-term gradient magnitude independent of T.
* alpha balances hard labels vs teacher soft targets.
* Feature distillation pairs intermediate maps of student and teacher at the
  same spatial size (8x8 "mid" and 4x4 "final" hints):
    - "fitnet"    : 1x1 conv + BN regressor maps student channels -> teacher channels,
                    then MSE between per-sample normalised maps (FitNets, Romero et al., 2015)
    - "attention" : channel-free spatial attention maps, L2 distance
                    (Attention Transfer, Zagoruyko & Komodakis, 2017)

Setting alpha=0 and beta=0 reduces to ordinary training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax_with_temperature(logits, T):
    """Temperature-scaled softmax: p_i = exp(z_i / T) / sum_j exp(z_j / T)."""
    return F.softmax(logits / T, dim=1)


def soft_target_loss(student_logits, teacher_logits, T):
    """KL(teacher || student) on temperature-softened distributions, scaled by T^2."""
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = softmax_with_temperature(teacher_logits, T)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)


def attention_map(f, p=2):
    """Spatial attention: mean over channels of |f|^p, flattened and L2-normalised."""
    return F.normalize(f.pow(p).mean(1).flatten(1), dim=1)


class FeatureRegressor(nn.Module):
    """1x1 conv + BN that projects a student feature map into the teacher feature space."""

    def __init__(self, cin, cout):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return self.proj(x)


class Distiller(nn.Module):
    """Wraps (student, frozen teacher, feature regressors) and computes the KD loss.

    Only the student and the regressors are trainable; the teacher runs in eval
    mode under torch.no_grad().
    """

    def __init__(self, student, teacher, T=4.0, alpha=0.9, beta=1.0, feat_loss="fitnet",
                 hint_layers=(0, 1), input_size=32):
        super().__init__()
        self.student = student
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.T, self.alpha, self.beta = T, alpha, beta
        self.feat_loss = feat_loss
        self.hint_layers = list(hint_layers)

        # Discover feature shapes with a dummy forward pass and build regressors
        self.regressors = nn.ModuleList()
        if beta > 0 and feat_loss == "fitnet":
            dev = next(student.parameters()).device
            with torch.no_grad():
                x = torch.zeros(2, 3, input_size, input_size, device=dev)
                was_training = student.training
                student.eval()
                _, fs = student(x, return_features=True)
                _, ft = teacher(x, return_features=True)
                student.train(was_training)
            for i in self.hint_layers:
                assert fs[i].shape[-2:] == ft[i].shape[-2:], "hint feature maps must share spatial size"
                self.regressors.append(FeatureRegressor(fs[i].shape[1], ft[i].shape[1]).to(dev))

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()  # teacher BN statistics must never update
        return self

    def feature_loss(self, fs, ft):
        loss = 0.0
        for j, i in enumerate(self.hint_layers):
            s, t = fs[i].float(), ft[i].float()
            if self.feat_loss == "fitnet":
                # normalise each map (zero mean / unit variance per sample) so the
                # hint loss is scale-invariant and comparable across layers
                r = self.regressors[j](fs[i]).float()
                loss = loss + F.mse_loss(F.layer_norm(r, r.shape[1:]), F.layer_norm(t, t.shape[1:]))
            elif self.feat_loss == "attention":
                loss = loss + (attention_map(s) - attention_map(t)).pow(2).mean()
            else:
                raise ValueError(self.feat_loss)
        return loss

    def forward(self, x, y):
        need_feats = self.beta > 0
        with torch.no_grad():
            t_out = self.teacher(x, return_features=need_feats)
        s_out = self.student(x, return_features=need_feats)
        (t_logits, ft) = t_out if need_feats else (t_out, None)
        (s_logits, fs) = s_out if need_feats else (s_out, None)

        ce = F.cross_entropy(s_logits, y)
        kd = soft_target_loss(s_logits.float(), t_logits.float(), self.T) if self.alpha > 0 else s_logits.new_zeros(())
        feat = self.feature_loss(fs, ft) if need_feats else s_logits.new_zeros(())
        loss = (1 - self.alpha) * ce + self.alpha * kd + self.beta * feat
        parts = {"ce": ce.item(), "kd": float(kd), "feat": float(feat)}
        return loss, s_logits, parts
