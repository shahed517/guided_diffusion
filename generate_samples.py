import torch
import matplotlib.pyplot as plt
from SDE import SDEDiffusion  # Assumes your logic is in SDE.py
from models import MNISTClassifier, PowerfulUNet  # Import the U-Net we built earlier

def generate_unconditional_samples(model_path="/home/shahed/guided_diffusion/weights/mnist_model.pth", num_samples=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize the architecture
    estimator = PowerfulUNet().to(device)
    
    # 2. Initialize SDE and load the saved weights
    sde = SDEDiffusion(estimator=estimator)
    sde.load_weights(model_path)
    estimator.eval()

    # 3. Create initial noise (z ~ N(0, I))
    # MNIST is 1 channel, 28x28
    z = torch.randn(num_samples, 1, 28, 28).to(device)

    # 4. Run Reverse Diffusion
    print(f"Generating {num_samples} samples...")
    with torch.no_grad():
        # Using stoc=True often yields better results for SDEs
        samples = sde.reverse_diffusion(z, n_timesteps=500, stoc=1)

    # 5. Post-process: Scale from [-1, 1] back to [0, 1] for plotting
    samples = (samples + 1.0) / 2.0
    samples = samples.clamp(0.0, 1.0).cpu().numpy()

    # 6. Plotting in a grid
    fig, axes = plt.subplots(2, 10, figsize=(15, 4))
    fig.suptitle("Randomly Generated MNIST Digits", fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        # MNIST is grayscale, so we use cmap='gray'
        ax.imshow(samples[i, 0], cmap='gray')
        ax.axis('off')
        ax.set_title(f"Sample {i+1}")

    plt.tight_layout()
    plt.savefig("/home/shahed/guided_diffusion/samples/generated_uncond_samples.png")
    print("Samples saved to generated_samples.png")
    plt.show()



def generate_conditional_samples(
    model_path="/home/shahed/guided_diffusion/weights/mnist_model.pth",
    classifier_path="/home/shahed/guided_diffusion/weights/mnist_classifier.pth",
    target_class=7,
    guidance_scale=4.0,
    num_samples=20,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --------------------------------------------------
    # 1. Load diffusion model
    # --------------------------------------------------
    estimator = PowerfulUNet().to(device)
    sde = SDEDiffusion(estimator=estimator)
    sde.load_weights(model_path)

    estimator.eval()
    for p in estimator.parameters():
        p.requires_grad_(False)

    # --------------------------------------------------
    # 2. Load classifier
    # --------------------------------------------------
    classifier = MNISTClassifier().to(device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.eval()

    for p in classifier.parameters():
        p.requires_grad_(False)

    # --------------------------------------------------
    # 3. Initial noise
    # --------------------------------------------------
    z = torch.randn(num_samples, 1, 28, 28).to(device)

    # --------------------------------------------------
    # 4. Guided reverse diffusion (NO torch.no_grad!)
    # --------------------------------------------------
    print(
        f"Generating {num_samples} samples "
        f"conditioned on class {target_class} "
        f"(guidance scale = {guidance_scale})..."
    )

    samples = sde.guided_reverse_diffusion(
        z=z,
        classifier=classifier,
        target_class=target_class,
        guidance_scale=guidance_scale,
        n_timesteps=500,
        stoc=1,
    )

    # --------------------------------------------------
    # 5. Post-process to [0, 1]
    # --------------------------------------------------
    samples = (samples + 1.0) / 2.0
    samples = samples.clamp(0.0, 1.0).cpu().numpy()

    # --------------------------------------------------
    # 6. Plot
    # --------------------------------------------------
    fig, axes = plt.subplots(2, 10, figsize=(15, 4))
    fig.suptitle(
        f"MNIST Samples Conditioned on Class {target_class}",
        fontsize=16
    )

    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i, 0], cmap="gray")
        ax.axis("off")
        ax.set_title(f"Sample {i+1}")

    plt.tight_layout()
    save_path = (
        f"/home/shahed/guided_diffusion/samples/generated_cond_samples.png"
    )
    plt.savefig(save_path)
    print(f"Samples saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    model_path = "/home/shahed/guided_diffusion/weights/mnist_model_60.pth"
    classifier_path = "/home/shahed/guided_diffusion/weights/mnist_classifier.pth"
    generate_conditional_samples(model_path=model_path, classifier_path=classifier_path, target_class=3, guidance_scale=5)
    # generate_unconditional_samples(model_path=model_path)