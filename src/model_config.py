config_dict = {
    'sgpt': {
        "backbone": "cnn",
        "n_epochs": 200,
        "lr": 1e-4,
        "dropout": 0,
        "weight_decay": 0,
        "image_size": 224,
        "cond_dim": 7,
        "hidden_dim": 512,
        "common_dim": 128,
        "latent_dim": 128,
        "split": "sgpt"
    },

    'mech': {
        "latent_dim": 128,
        "hidden_dim": 128,
        "num_layers": 3
    },

    'prior': {
        "dim": 128,
        "depth": 6,
        "dim_head": 64,
        "heads": 8,
        "image_embed_dim": 128,
        "timesteps": 1000,
        "cond_drop_prob": 0.2
    },

    'decoder': {
        "dim": 128,
        "image_embed_dim": 128,
        "cond_dim": 128,
        "channels": 3,
        "dim_mults": (1, 2, 4, 8),
        "image_size": 224,
        "timesteps": 1000,
        "image_cond_drop_prob": 0.1,
        "text_cond_drop_prob": 0.5
    },

    'weight_paths': {
        "sgpt": "../models/pretrained/sgpt_gen.pth",
        "mech": "../models/mech/mech.pth",
        "prior": "../models/prior/prior.pth",
        "decoder": "../models/decoder/decoder.pth"
    }
}