import torch
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
from dalle2_pytorch import DiffusionPrior


def exists(val):
    return val is not None


def l2norm(t):
    return F.normalize(t, dim = -1)


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class DiffusionPriorEmbed(DiffusionPrior):
    @torch.no_grad()
    @eval_decorator
    def sample(
            self,
            text_embed,
            text_encodings=None,
            num_samples_per_batch=2,
            cond_scale=1.,
            timesteps=None
    ):
        timesteps = default(timesteps, self.sample_timesteps)

        # in the paper, what they did was
        # sample 2 image embeddings, choose the top 1 similarity, as judged by CLIP

        text_embed = repeat(text_embed, 'b d -> (b r) d', r=num_samples_per_batch)
        batch_size = text_embed.shape[0]
        image_embed_dim = self.image_embed_dim

        text_cond = dict(text_embed=text_embed)

        if self.condition_on_text_encodings:
            text_cond = {**text_cond, 'text_encodings': text_encodings}

        image_embeds = self.p_sample_loop((batch_size, image_embed_dim), text_cond=text_cond, cond_scale=cond_scale,
                                          timesteps=timesteps)

        # retrieve original unscaled image embed

        text_embeds = text_cond['text_embed']

        text_embeds = rearrange(text_embeds, '(b r) d -> b r d', r=num_samples_per_batch)
        image_embeds = rearrange(image_embeds, '(b r) d -> b r d', r=num_samples_per_batch)

        text_image_sims = einsum('b r d, b r d -> b r', l2norm(text_embeds), l2norm(image_embeds))
        top_sim_indices = text_image_sims.topk(k=1).indices

        top_sim_indices = repeat(top_sim_indices, 'b 1 -> b 1 d', d=image_embed_dim)

        top_image_embeds = image_embeds.gather(1, top_sim_indices)
        return rearrange(top_image_embeds, 'b 1 d -> b d')


if __name__ == "__main__":
    pass