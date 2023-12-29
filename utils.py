import json
import sys, ipdb
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os


def read_config(file_path: str) -> dict:
    file = open(file_path, "r")
    try:
        try:
            df = json.load(file)
            # validate(df)
            return df
        except json.decoder.JSONDecodeError:
            print("Please enter all field values.")

    finally:
        file.close()


# token allocation vis
def plot_allocation(token, n_bins=10, save_fig=True):
    plt.hist(token, n_bins, density=0, color="green", alpha=0.7)

    plt.xlabel("Tokens")
    plt.ylabel("Agent number")
    plt.title("Token allocation")

    if save_fig:
        plt.savefig(os.path.join("vis", "plot.png"))
    else:
        plt.show()


def animate_plot_allocation(token, n_bins=10, save_fig=True):
    def prepare_animation(bar_container):
        def animate(frame_number):
            # simulate new data coming in
            data = token[frame_number]
            n, _ = np.histogram(data, n_bins)
            for count, rect in zip(n, bar_container.patches):
                rect.set_height(count)
            return bar_container.patches

        return animate

    fig, ax = plt.subplots()
    _, _, bar_container = ax.hist(
        token[0], n_bins, lw=1, ec="yellow", fc="green", alpha=0.5
    )
    ax.set_ylim(top=55)  # set safe limit to ensure that all data is visible.

    anim = animation.FuncAnimation(
        fig,
        prepare_animation(bar_container),
        frames=len(token) - 1,
        interval=200,
        blit=True,
        repeat=True,
    )

    if save_fig:
        anim.save(os.path.join("vis", "plt.mp4"), writer="ffmpeg", fps=30)
    else:
        plt.show()
