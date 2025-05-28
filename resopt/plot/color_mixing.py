import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys

def polar2z(r, theta):
    return r * np.exp(1j * theta)

def z2polar(z):
    return (np.abs(z), np.angle(z))

def mix_colors(c_list, vmin=0, vmax=1):
    c_list_z = []
    c_list_v = []

    for c in c_list:
        c_norm = (c - vmin) / (vmax - vmin)

        h, s, v = mcolors.rgb_to_hsv(c_norm)
        h_polar = h * 2 * np.pi
        s_polar = s

        c_list_z.append(polar2z(s_polar, h_polar))
        c_list_v.append(v)

    try:
        z_mean = np.average(c_list_z, weights=c_list_v)
    except ZeroDivisionError:
        z_mean = 0+0j
    v = min(1., np.sum(c_list_v))

    s_polar, h_polar = z2polar(z_mean)
    h = (h_polar % (2 * np.pi)) / (2 * np.pi)
    s = s_polar

    r, g, b = mcolors.hsv_to_rgb([h, s, v])
    c_out = np.array([r,g,b])

    return c_out * (vmax - vmin) + vmin

if __name__ == "__main__":
    c1 = np.array([1, 0, 0])
    c2 = np.array([0, 1, 0])
    c3 = np.array([0, 0, 1])

    cmix = mix_colors([c1, c2, c3]).astype(np.uint8)

    fig, ax = plt.subplots(2,2)

    ax[0,0].imshow(np.array([[c1]]))
    ax[0,1].imshow(np.array([[c2]]))
    ax[1,0].imshow(np.array([[c3]]))
    ax[1,1].imshow(np.array([[cmix]]))

    plt.show()




