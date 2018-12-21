import matplotlib.pyplot as plt


def suplabel(axis, label, label_prop=None,
             labelpad=5,
             ha='center', va='center'):
    ''' Add super ylabel or xlabel to the figure
    Similar to matplotlib.suptitle
    axis       - string: "x" or "y"
    label      - string
    label_prop - keyword dictionary for Text
    labelpad   - padding from the axis (default: 5)
    ha         - horizontal alignment (default: "center")
    va         - vertical alignment (default: "center")
    '''
    fig = plt.gcf()
    xmin = []
    ymin = []
    for ax in fig.axes:
        xmin.append(ax.get_position().xmin)
        ymin.append(ax.get_position().ymin)
    xmin, ymin = min(xmin), min(ymin)
    dpi = fig.dpi
    if axis.lower() == "y":
        rotation = 90.
        x = xmin - float(labelpad) / dpi
        y = 0.5
    elif axis.lower() == 'x':
        rotation = 0.
        x = 0.5
        y = ymin - float(labelpad) / dpi
    else:
        raise Exception("Unexpected axis: x or y")
    if label_prop is None:
        label_prop = dict()

    plt.text(
        x, y, label, rotation=rotation, transform=fig.transFigure, ha=ha, va=va, **label_prop
    )


def make_plot(data, bl_nums, u_points, bl_min, bl_max,
              plot_perfect=True, xscale_log=True, legend=True, supx=r"$\omega$", supy="Var(V)",
              normalise_at=None, xquant='omega'):
    fig, ax = plt.subplots(
        2, 4, sharex=True, sharey=True,
        subplot_kw={"xscale": 'log' if xscale_log else "linear", "yscale": 'log'},
        gridspec_kw={"hspace": .05, "wspace": .05},
        squeeze=False, figsize=(15, 6)
    )

    if xquant != "omega" and plot_perfect:
        raise ValueError("if plotting the perfect curve, require xquant='omega'")

    for axx in ax.flatten():
        axx.grid(True)

        if plot_perfect:
            axx.plot(data[xquant], data['perfect'], color='k', label='Perfectly Redundant')

    for row, kind in enumerate(['lin', 'log']):
        ax[row, 0].set_ylabel(kind, fontsize=15)

        for col, n in enumerate(bl_nums):
            if row == 0:
                ax[row, col].set_title(r"$N_{\rm bl} = %s$" % (n), fontsize=15)

            if kind == "lin":
                ax[row, col].text(
                    0.65, 0.8, r"$\Delta_u = %.1f$" % ((bl_max - bl_min) / (n - 1)),
                    transform=ax[0, col].transAxes, fontsize=15
                )
            else:
                ax[row, col].text(
                    0.1, 0.1,
                    r"$\Delta = u\times%.1e$" % ((bl_max / bl_min) ** (1. / (n - 1)) - 1),
                    transform=ax[row, col].transAxes, fontsize=15
                )

            for uu in u_points:
                this = data[kind][f'n={n} u={uu}']

                if normalise_at is not None:
                    if plot_perfect:
                        this *= data['perfect'][normalise_at] / this[normalise_at]
                    else:
                        this *= 1 / this[normalise_at]

                ax[row, col].plot(data[xquant], this, label=f'u={uu}')

    if legend:
        ax[0, 0].legend()

    suplabel('x', supx, label_prop={"fontsize": 17})
    suplabel('y', supy, label_prop={"fontsize": 17})
