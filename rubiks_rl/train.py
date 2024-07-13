import math

import torch
from gymnasium import spaces
from positional_encodings.torch_encodings import PositionalEncoding1D
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import RecurrentPPO
import torch.backends


from rubiks_rl.environment import RubiksCube


def next_power_of_2(x: int) -> int:
    return pow(2, math.ceil(math.log(x) / math.log(2)))


class AttentionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # 6 faces i.e. 6 classes
        n_heads = 8
        n_emb = next_power_of_2(int(math.ceil(6**0.25)))
        self.embeddings = torch.nn.Embedding(6, n_heads * n_emb)
        self.positional_encoding = PositionalEncoding1D(n_heads * n_emb)
        self.multihead_attn = torch.nn.MultiheadAttention(
            embed_dim=n_heads * n_emb, num_heads=n_heads, batch_first=True
        )
        self.linear = torch.nn.Linear(864, features_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Something is casting to float...ugh
        x = x.int()
        x = x.flatten(start_dim=1)
        x = self.embeddings(x)
        x = self.positional_encoding(x)
        # self-attention
        x = self.multihead_attn(x, x, x, need_weights=False)[0]
        x = x.flatten(start_dim=1)
        x = self.linear(x)
        return x


def main():
    vec_env = make_vec_env(lambda: RubiksCube(shuffle=True), 16)
    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        policy_kwargs={"features_extractor_class": AttentionExtractor},
        verbose=2,
    )
    mode = "default"
    if torch.cuda.is_available():
        mode = "reduce-overhead"
        model.policy = model.policy.to(device="cuda")

    model.policy.compile(mode=mode)
    model.learn(total_timesteps=250_000, progress_bar=True)

    model.save("rubiks")

    del model  # remove to demonstrate saving and loading

    model = RecurrentPPO.load("rubiks")

    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        print("Reward:", rewards)
        print(obs)


if __name__ == "__main__":
    main()
