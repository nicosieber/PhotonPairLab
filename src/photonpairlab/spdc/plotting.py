import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator

from photonpairlab.spdc.analysis.fitting import gaussian
from photonpairlab.spdc.simulation.results import SPDCResults
from photonpairlab.spdc.analysis.spectral_analyser import SpectralAnalyzer
from photonpairlab.spdc.analysis.two_mode_hom_results import TwoModeHOMResults

# Modern light-mode chart palette: neutral chart chrome plus a colorblind-validated
# categorical series order (fixed assignment order -- never cycle/reorder per-plot).
_SURFACE = "#fcfcfb"
_GRID = "#e1e0d9"
_BASELINE = "#c3c2b7"
_PRIMARY_INK = "#0b0b0b"
_SECONDARY_INK = "#52514e"
_MUTED_INK = "#898781"

_SERIES_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]


def _style_axes(ax, grid=True):
    """Apply the shared modern chart style: light surface, recessive gridlines/spines, muted ink."""
    ax.set_facecolor(_SURFACE)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(True, color=_GRID, linewidth=0.9)
    else:
        ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_BASELINE)
    ax.spines["bottom"].set_color(_BASELINE)
    ax.tick_params(colors=_MUTED_INK, labelsize=9)
    ax.xaxis.label.set_color(_SECONDARY_INK)
    ax.yaxis.label.set_color(_SECONDARY_INK)
    ax.title.set_color(_PRIMARY_INK)


def _style_legend(ax):
    legend = ax.legend(frameon=False, labelcolor=_SECONDARY_INK)
    return legend


class SPDC_Plotter:
    #: Colorblind-validated categorical series colors, in fixed assignment order.
    SERIES_COLORS = _SERIES_COLORS

    def __init__(self,  results: SPDCResults):
        self.results = results

    @staticmethod
    def style_axes(ax, grid=True):
        """
        Apply the shared modern chart style (light surface, recessive gridlines/spines,
        muted ink) to an arbitrary matplotlib ``Axes``.

        Exposed as a public helper so one-off plots (e.g. in a notebook) can match the
        look of the plots produced by this class without duplicating the style constants.
        """
        _style_axes(ax, grid=grid)

    def plot_schmidt_coefficients(self, n_coefficients=20, fig=None, ax=None, font_size=12):
        """
        Plot the Schmidt-coefficient histogram from this result's Schmidt decomposition.

        Parameters
        ----------
        n_coefficients : int
            Number of leading Schmidt coefficients to show (clipped to however many exist).
        fig, ax : matplotlib Figure/Axes, optional
            Existing figure/axes to draw into (e.g. one panel of a larger figure you're
            composing yourself). Both or neither must be given.

        Returns
        -------
        (fig, ax)

        See Also
        --------
        plot_signal_idler_spectra : the signal/idler marginal-spectra plot, previously
            bundled into this method as a second panel -- now separate so callers can use
            either, both, or neither, in whatever layout they want.
        """
        analyzer = SpectralAnalyzer(self.results)
        s_vals, Purity, _ = analyzer.schmidt_decomposition()

        if fig is None and ax is None:
            fig, ax = plt.subplots(facecolor=_SURFACE)
        elif fig is not None and ax is not None:
            pass
        else:
            raise ValueError("Both fig and ax must be either None or provided together.")

        n = min(n_coefficients, len(s_vals))
        ax.bar(np.arange(n), s_vals[:n], align="center", color=_SERIES_COLORS[0], width=0.7, zorder=3)
        ax.set_ylabel("Schmidt Coefficients", fontsize=font_size)
        ax.set_title(f"Schmidt Decomposition of the JSA - Resulting purity: {round(Purity, 2)}", fontsize=font_size)
        _style_axes(ax)

        return fig, ax

    def plot_signal_idler_spectra(self, fitting_function=gaussian, fig=None, ax=None, font_size=12):
        """
        Plot the signal/idler marginal spectra (the JSI summed over the other photon's
        wavelength), each with ``fitting_function`` fitted and overlaid.

        Parameters
        ----------
        fitting_function : callable
            Passed through to :meth:`SpectralAnalyzer.get_signal_idler_fits`.
        fig, ax : matplotlib Figure/Axes, optional
            Existing figure/axes to draw into. Both or neither must be given.

        Returns
        -------
        (fig, ax)
        """
        analyzer = SpectralAnalyzer(self.results)
        signal_fit, idler_fit, (signal_wavelenghts, signal_intensities), (idler_wavelengths, idler_intensities) = analyzer.get_signal_idler_fits(fitting_function)

        if fig is None and ax is None:
            fig, ax = plt.subplots(facecolor=_SURFACE)
        elif fig is not None and ax is not None:
            pass
        else:
            raise ValueError("Both fig and ax must be either None or provided together.")

        signal_color, idler_color = _SERIES_COLORS[0], _SERIES_COLORS[1]

        ax.plot(signal_wavelenghts, signal_intensities, "o", color=signal_color, markersize=5, zorder=3)
        ax.plot(signal_wavelenghts, fitting_function(signal_wavelenghts, *signal_fit), linestyle="--", color=signal_color, linewidth=2, zorder=3)
        ax.plot(idler_wavelengths, idler_intensities, "^", color=idler_color, markersize=5, zorder=3)
        ax.plot(idler_wavelengths, fitting_function(idler_wavelengths, *idler_fit), linestyle="--", color=idler_color, linewidth=2, zorder=3)

        ax.set_xlim(left=np.amin(signal_wavelenghts), right=np.amax(signal_wavelenghts))
        ax.set_xlabel("wavelength (nm)")
        ax.set_ylabel("normalized amplitude", fontsize=font_size)
        ax.set_title("JSI Profiles", fontsize=font_size)
        ax.legend(["signal", "fit: signal", "idler", "fit: idler"], frameon=False, labelcolor=_SECONDARY_INK)
        _style_axes(ax)

        return fig, ax

    def plot_result(self, key="JSA", fig=None, ax=None, font_size=12, color_map=cm.viridis, colorbar=True): # type: ignore
        number_ticklabels = 5

        signal_wavelengths = self.results.SignalWavelengths * 1e9
        idler_wavelengths = self.results.IdlerWavelengths * 1e9

        if fig is None and ax is None:
            fig, axs = plt.subplots(1, 1, sharex=True, constrained_layout=False, facecolor=_SURFACE)
        elif fig is not None and ax is not None:
            axs = ax
            fig = fig
        else:
            raise ValueError("Both fig and ax must be either None or provided together.")

        # JSA is complex (phase included); plot its magnitude like the other (already-real) keys.
        PLOT_KEY_HANDLER = {
            "Pump": self.results.Pump,
            "Phase": self.results.Phase,
            "JSI": self.results.JSI,
            "JSA": np.abs(self.results.JSA),
        }
        extent = (
            float(signal_wavelengths.min()),
            float(signal_wavelengths.max()),
            float(idler_wavelengths.min()),
            float(idler_wavelengths.max()),
        )
        im = axs.imshow(
            PLOT_KEY_HANDLER[key] / np.amax(PLOT_KEY_HANDLER[key]),
            cmap=color_map,
            extent=extent,
            origin='lower' # or 'upper' if you want to flip y
            )
        im.set_interpolation("bilinear")

        axs.set_xlabel("signal wavelength (nm)", fontsize=font_size)
        axs.set_ylabel("idler wavelength (nm)", fontsize=font_size)

        axs.xaxis.set_major_locator(MaxNLocator(number_ticklabels))
        axs.yaxis.set_major_locator(MaxNLocator(number_ticklabels))
        _style_axes(axs, grid=False)
        # imshow fills the whole axes frame; keep a thin neutral border instead of the
        # (invisible, grid-only) hairline spines used elsewhere.
        for spine in axs.spines.values():
            spine.set_visible(True)
            spine.set_color(_BASELINE)

        if colorbar:
            cbar = fig.colorbar(im, ax=axs, fraction=0.046, pad=0.04)
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(colors=_MUTED_INK, labelsize=9)

        return fig, axs

    @staticmethod
    def plot_poling_profile(crystal, fig=None, ax=None, font_size=12):
        """
        Plot target vs. actual field-amplitude buildup along a crystal's poling pattern (top
        panel), with the domain structure itself shown as a bar-code strip below.

        Works for any crystal on which ``generate_poling(...)`` has been called, regardless of
        poling mode (periodic, constant/unpoled, or apodized ``subcoh``) or strategy (QPM/APM):
        ``Crystal.generate_poling()`` always populates ``target_amplitude``/``actual_amplitude``
        alongside ``poling_pattern``/``z``. For non-apodized modes, "target" is the ideal,
        fully-efficient uniform envelope (see ``PhaseMatchingStrategy.uniform_target``), so the
        curves show how closely a plain periodic grating (or, for an unpoled crystal, how poorly)
        tracks that ideal buildup. For a plain periodic/constant grating specifically, expect
        ``actual`` to run ~4/pi above ``target`` -- see
        ``PhaseMatchingStrategy.uniform_target`` for why (expected, not a bug).

        Parameters
        ----------
        crystal : Crystal
            A crystal on which ``generate_poling(...)`` has already been called.
        fig, ax : matplotlib Figure/(Axes, Axes), optional
            Existing figure and a ``(field_ax, pattern_ax)`` pair of axes to draw into. Both or
            neither must be given.

        Returns
        -------
        (fig, (field_ax, pattern_ax))
        """
        if crystal.poling_pattern is None:
            raise ValueError("crystal.generate_poling(...) must be called before plotting.")

        if fig is None and ax is None:
            fig, (field_ax, pattern_ax) = plt.subplots(
                2, 1, sharex=True, facecolor=_SURFACE,
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
            )
        elif fig is not None and ax is not None:
            field_ax, pattern_ax = ax
        else:
            raise ValueError("Both fig and ax must be either None or provided together.")

        z = crystal.z
        target = np.abs(crystal.target_amplitude)
        actual = np.abs(crystal.actual_amplitude)
        norm = np.amax(target) or 1.0

        field_ax.plot(z, actual / norm, color=_SERIES_COLORS[0], linewidth=2, label="Actual Field", zorder=3)
        field_ax.plot(z, target / norm, color=_SERIES_COLORS[1], linewidth=2, label="Target", zorder=3)
        field_ax.set_ylabel("field amplitude", fontsize=font_size)
        _style_axes(field_ax)
        _style_legend(field_ax)

        pattern_ax.imshow(
            crystal.poling_pattern[None, :], aspect="auto", cmap="gray",
            extent=(float(z.min()), float(z.max()), 0, 1),
        )
        pattern_ax.set_yticks([])
        pattern_ax.set_xlabel("position (m)", fontsize=font_size)
        _style_axes(pattern_ax, grid=False)
        pattern_ax.spines["left"].set_visible(False)

        return fig, (field_ax, pattern_ax)

    @staticmethod
    def plot_hom_dip(results, fig=None, ax=None, font_size=12):
        """
        Plot one or more Hong-Ou-Mandel coincidence-probability dips vs. relative delay.

        Parameters
        ----------
        results : TwoModeHOMResults or Mapping[str, TwoModeHOMResults]
            A single :class:`~photonpairlab.spdc.analysis.two_mode_hom_results.TwoModeHOMResults`
            (from :meth:`~photonpairlab.spdc.analysis.hom_analyser.HOMAnalyzer.compute_two_mode_HOM`),
            or a ``{label: TwoModeHOMResults}`` mapping to overlay multiple dips (e.g. comparing
            sources).
        fig, ax : matplotlib Figure/Axes, optional
            Existing figure/axes to draw into. Both or neither must be given.

        Returns
        -------
        (fig, ax)

        Notes
        -----
        This is a ``staticmethod``: it plots whichever ``TwoModeHOMResults`` are passed in,
        independent of any ``SPDC_Plotter`` instance's bound ``results``.
        """
        if isinstance(results, TwoModeHOMResults):
            results = {"HOM dip": results}

        if fig is None and ax is None:
            fig, ax = plt.subplots(facecolor=_SURFACE)
        elif fig is not None and ax is not None:
            pass
        else:
            raise ValueError("Both fig and ax must be either None or provided together.")

        for (label, hom_result), color in zip(results.items(), _SERIES_COLORS):
            ax.plot(hom_result.tau_fs, hom_result.coincidence_probabilities, label=label, color=color, linewidth=2, zorder=3)

        ax.set_xlabel("relative delay (fs)", fontsize=font_size)
        ax.set_ylabel("coincidence probability", fontsize=font_size)
        _style_axes(ax)
        if len(results) > 1:
            _style_legend(ax)

        return fig, ax
