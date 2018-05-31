"""
Functions to make diagnosis plots.
"""

import corner
import matplotlib.pyplot as plt


def plot_walkers(samples, nburnin=None, labels=None):
    """Plot the walkers to check if they are burning in."""

    # Check the length of the label list.
    if labels is None:
        if samples.shape[0] != len(labels):
            raise ValueError("Not correct number of labels.")

    # Cycle through the plots.
    for s, sample in enumerate(samples):
        fig, ax = plt.subplots()
        for walker in sample.T:
            ax.plot(walker, alpha=0.1)
        ax.set_xlabel('Steps')
        if labels is not None:
            ax.set_ylabel(labels[s])
        if nburnin is not None:
            ax.axvline(nburnin, ls=':', color='r')


def plot_corner(samples, labels=None, quantiles=[0.16, 0.5, 0.84]):
    """Plot the corner plot to check for covariances."""
    corner.corner(samples, labels=labels,
                  quantiles=quantiles, show_titles=True)
