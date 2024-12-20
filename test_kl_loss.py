"""Tests just the KL loss."""
import torch
import torch.distributions as dist
import torch.optim as optim
import tqdm

from train import load_training_data


def categorical_kl_divergence(phi: torch.Tensor) -> torch.Tensor:
    # phi is logits of shape [B, N, K] where B is batch, N is number of categorical distributions, K is number of classes
    B, N, K = phi.shape
    phi = phi.view(B * N, K)
    q = dist.Categorical(logits=(phi + 1e-20))
    p = dist.Categorical(
        probs=torch.full((B * N, K), 1.0 / K)
    )  # uniform bunch of K-class categorical distributions
    kl = dist.kl.kl_divergence(q, p)  # kl is of shape [B*N]
    return kl.view(B, N)


def main():
    N = 1
    K = 2
    max_steps = 2000
    initial_learning_rate = 1e-5
    batch_size = 100
    training_images = load_training_data()

    train_dataset = torch.utils.data.DataLoader(
        dataset=training_images, batch_size=batch_size, shuffle=True
    )

    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(28 * 28, N * K))
    optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate, momentum=0.0)

    step = 0
    progress_bar = tqdm.tqdm(total=max_steps, desc="Training")
    while step < max_steps:
        for (
            data
        ) in (
            train_dataset
        ):  # x should be a batch of torch.Tensor spectrograms, of shape [B, F, T]
            x, labels = data

            phi = model(x).view(-1, N, K)
            step += 1

            kl_loss = torch.mean(torch.sum(categorical_kl_divergence(phi), dim=1))
            kl_loss.backward()
            optimizer.step()

            if step == 1:
                breakpoint()
            if step == 400:
                breakpoint()
            progress_bar.set_description(
                f"Training | KL loss = {kl_loss:.7f} / phi.mean() = {phi.exp().mean()}"
            )
            progress_bar.update(1)
            kl_loss = torch.mean(torch.sum(categorical_kl_divergence(phi), dim=1))


if __name__ == "__main__":
    main()
