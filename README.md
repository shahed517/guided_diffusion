The repository contains the code to a small side-project to generate handwritten digits using diffusion models. I used the MNIST dataset to train my own UNet-based diffusion model following the denoising process outlined in the VP-SDE paper.
The first objective was to generate handwritten digits unconditionally, which was achieved following a simple application of the reverse diffusion procedure. Some representative results achieved after 500 denoising steps are as follows:

![generated unconditional samples](https://github.com/shahed517/guided_diffusion/blob/main/samples/generated_uncond_samples.png)

Next a digit classifer was trained to classify digits after random noise injection, similar to the forward diffusion process. For a specific class (e.g. digit 8) the gradient of the trained classifier was calculated using pytorch and this was added to the estimated score with some weighting during reverse diffusion. This resulted in the following conditional samples:

![generated unconditional samples](https://github.com/shahed517/guided_diffusion/blob/main/samples/generated_cond_samples.png)

References:
1. Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations (ICLR). 

