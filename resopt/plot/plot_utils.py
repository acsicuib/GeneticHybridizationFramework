import numpy as np
from resopt.param.default import OBJ_LIST, OBJ_DESCRIPTION

class LegendPicker:
    def __init__(self, points):
        self.points = points
        self.fig = next(iter(points.values())).figure
        self.fig.canvas.mpl_connect('pick_event', self)

    def __call__(self, event):
        legtext = event.artist
        name = legtext.get_text()
        vis = not self.points[name].get_visible()
        self.points[name].set_visible(vis)
        if vis:
            legtext.set_alpha(1.0)
        else:
            legtext.set_alpha(0.2)

        self.fig.canvas.draw()

class GenerationPicker:
    def __init__(self, generation, points):
        self.prev_he = False
        self.prev_he_gen = 0

        self.generation = generation
        self.sizes = np.array([30 for _ in generation])
        self.points = points

        self.points.figure.canvas.mpl_connect('motion_notify_event', self)

    def __call__(self, event):
        if event.inaxes is not None and \
                hasattr(event.inaxes, "hover_event") and \
                event.ydata is not None:
            self.prev_he = True
            gen = int(round(event.ydata))
            #gen = int(event.ydata * self.generation[-1]) + 1
            if gen != self.prev_he_gen:
                print(gen)
                if self.prev_he_gen == 0:
                    for i in range(len(self.sizes)):
                        curr_gen = self.generation[i]
                        if curr_gen != gen:
                            self.sizes[i] = 0
                        else:
                            self.sizes[i] = 30
                else:
                    # Set current selected self.generation to 1
                    idx_first = self.generation.index(gen)
                    idx_next = len(self.generation) \
                            if gen == self.generation[-1] \
                            else self.generation.index(gen + 1)
                    for i in range(idx_first, idx_next):
                        self.sizes[i] = 30

                    # Set previous selected self.generation to 0
                    idx_first = self.generation.index(self.prev_he_gen)
                    idx_next = len(self.generation) \
                            if self.prev_he_gen == self.generation[-1] \
                            else self.generation.index(self.prev_he_gen + 1)
                    for i in range(idx_first, idx_next):
                        self.sizes[i] = 0

                self.prev_he_gen = gen

                self.points.set_sizes(self.sizes)
                self.points.figure.canvas.draw()
        elif self.prev_he:
            self.prev_he = False
            self.prev_he_gen = 0

            for i in range(len(self.sizes)):
                self.sizes[i] = 30

            self.points.set_sizes(self.sizes)
            self.points.figure.canvas.draw()

def get_recommended_ticks(o_min, o_max, integer=False):
    STEPS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.,
            2., 5., 10., 20., 50., 100.]
    DIVISIONS = 10
    for i in range(len(STEPS)):
        if o_max == o_min:
            return np.array([o_min])
        if STEPS[i] <= (o_max - o_min) / DIVISIONS < STEPS[i+1]:
            if integer and STEPS[i+1] < 1:
                step = 1
            else:
                step = STEPS[i+1]
            start = np.floor(o_min / step) * step
            end   = (np.ceil(o_max / step) + 1) * step
            return np.arange(start, end, step)

def getXYZLabel(xyz_idx, objectives):
    obj_idx = OBJ_LIST.index(objectives[xyz_idx])
    return OBJ_DESCRIPTION[obj_idx]

def linear_growth(x, xmax, vmin=0., vmax=1.):
    return ((vmax - vmin) / (xmax - 1)) * (x - 1) + vmin

def asymptotic_growth(x, growth=0.1, vmin=0., vmax=1.):
    fnorm = - np.exp(growth*(1-x)) + 1
    return fnorm * (vmax - vmin) + vmin
