import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import glob, os

for f in glob.glob(os.path.join(os.environ.get('CSBA_FONTS', ''), 'arial*.ttf')):
    font_manager.fontManager.addfont(f)

MM = 1/25.4
W2, W1 = 183*MM, 89*MM          # npj / Nature Portfolio double and single column; max height 170 mm
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 7, 'axes.titlesize': 7.5, 'axes.titleweight': 'normal', 'axes.labelsize': 7,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5, 'legend.frameon': False,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'lines.linewidth': 1.0, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlepad': 4,
    'savefig.dpi': 600, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.constrained_layout.h_pad': 0.04, 'figure.constrained_layout.w_pad': 0.04,
    'mathtext.default': 'regular', 'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold'})

OI = dict(black='#000000', orange='#E69F00', sky='#56B4E9', green='#009E73', yellow='#F0E442', blue='#0072B2',
          verm='#D55E00', purple='#CC79A7', grey='#8C8C8C')
GC = {'HC': OI['blue'], 'JHU_abnormal': OI['orange'], 'DCM': OI['verm'], 'Duke_abnormal': OI['purple'],
      'SG_compression': OI['green'], 'JHU_priorsurg': OI['grey']}
GL = {'HC': 'Healthy', 'JHU_abnormal': 'Center 1 abnormal', 'DCM': 'DCM', 'Duke_abnormal': 'CSpineSeg abnormal',
      'SG_compression': 'Asymptomatic compression', 'JHU_priorsurg': 'Prior surgery'}
CONTRAST_SHORT = {'Center 1 abnormal vs controls': 'Center 1 abnormal', 'DCM vs local controls (Centers 1-2)': 'DCM (Centers 1-2)',
                  'Duke abnormal vs Duke controls': 'CSpineSeg abnormal', 'Spine Generic compression vs controls': 'Asymptomatic compression'}
CONTRAST_COLOR = {'Center 1 abnormal vs controls': OI['orange'], 'DCM vs local controls (Centers 1-2)': OI['verm'],
                  'Duke abnormal vs Duke controls': OI['purple'], 'Spine Generic compression vs controls': OI['green']}


def panel(ax, s, x=-0.18, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='left')


def fmt_p(p):
    if p < 0.001: return '$\\it{P}$ < 0.001'
    return '$\\it{P}$ = ' + (f'{p:.1g}' if p < 0.01 else f'{p:.2g}' if p < 0.1 else f'{p:.2f}')


def save(fig, path):
    for e in ('png', 'pdf', 'tiff'):
        kw = dict(pil_kwargs={'compression': 'tiff_lzw'}) if e == 'tiff' else {}
        fig.savefig(f'{path}.{e}', dpi=600 if e != 'pdf' else None, **kw)
    plt.close(fig)


# display names for the clinical centers (internal keys stay Center1/Center2)
def relabel(s):
    if not isinstance(s, str): return s
    for a, b in [('Centers 1-2', 'JHU and MCW'), ('Centers 1–2', 'JHU and MCW'), ('Centers 1 and 2', 'JHU and MCW'),
                 ('Center 1', 'JHU'), ('Center 2', 'MCW'), ('Center1', 'JHU'), ('Center2', 'MCW')]:
        s = s.replace(a, b)
    return s


import matplotlib.text as _mtext
_set_text = _mtext.Text.set_text
_mtext.Text.set_text = lambda self, s: _set_text(self, relabel(s) if isinstance(s, str) else s)
