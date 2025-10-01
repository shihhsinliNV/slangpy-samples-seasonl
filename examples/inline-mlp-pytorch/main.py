# SPDX-License-Identifier: Apache-2.0

import torch
import timeit
import matplotlib.pyplot as plt
import slangpy as spy


def main():
    W, H = 512, 512
    device = spy.create_torch_device(type=spy.DeviceType.cuda)
    module = spy.Module.load_from_file(device, "inline_mlp_itensor.slang")

    # Params
    F, Hdim = 14, 32
    W1 = torch.randn((F, Hdim), dtype=torch.float32, device="cuda", requires_grad=True)
    b1 = torch.zeros((Hdim,), dtype=torch.float32, device="cuda", requires_grad=True)
    W2 = torch.randn((Hdim, 3), dtype=torch.float32, device="cuda", requires_grad=True)
    b2 = torch.zeros((3,), dtype=torch.float32, device="cuda", requires_grad=True)
    feature_grid = torch.randn((33, 33, F), dtype=torch.float32, device="cuda", requires_grad=True)

    # Load a target image as a torch tensor
    target = torch.from_numpy(plt.imread("media/jeep.jpg")).cuda()[:, :, :3].contiguous()
    # Convert from ByteTensor to FloatTensor & normalize to [0, 1]
    target = target.type(torch.float) / 255.0

    optim = torch.optim.Adam([W1, b1, W2, b2, feature_grid], lr=5e-3)

    img = torch.empty((W, H, 3), dtype=torch.float32, device="cuda", requires_grad=True)
    loss_fn = torch.nn.MSELoss()

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    def step(i):
        print(f"Iteration {i}")
        optim.zero_grad()

        nonlocal img

        start = timeit.default_timer()
        module.render_image(
            featureGrid=feature_grid,
            W1=W1,
            b1=b1,
            W2=W2,
            b2=b2,
            frameDim=[float(W), float(H)],
            grid_cell=spy.call_id(),
            _result=img,
        )
        end = timeit.default_timer()
        print(f"Forward pass: {end - start:.6f}s")

        loss = loss_fn(img, target)
        loss.backward()
        optim.step()

        if i % 50 == 0:
            ax1.clear()
            ax1.imshow(img.detach().cpu().clamp(0, 1).numpy())
            ax1.set_title("pred")
            ax2.clear()
            ax2.imshow(target.detach().cpu().numpy())
            ax2.set_title("target")

    import matplotlib.animation as animation

    ani = animation.FuncAnimation(fig, step, frames=4000, interval=10)
    writer = animation.FFMpegWriter(fps=30)
    ani.save("inline_mlp_pytorch.mp4", writer=writer)


if __name__ == "__main__":
    main()
