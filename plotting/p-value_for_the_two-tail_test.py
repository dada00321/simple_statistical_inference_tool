from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs

def plot():
	N = 150
	t_stat = -5*(5**0.5)
	t_crit = -2.132
	#--------------------------
	x = np.linspace(start = -4, stop = 4, num = int(1e5))
	hx = stats.t.pdf(x, df = N - 2)
	# Plot the density
	fig, ax = plt.subplots(num = 2, figsize = (10, 8))
	_ = plt.plot(x, hx, color = "black")
	_ = plt.margins(y = 0)
	_ = plt.ylim((0, 0.41))
	# Shade the probability p-value
	_ = plt.fill_between(x[x <= t_stat], hx[x <= t_stat], edgecolor = None,
	                 color = clrs.to_rgba(c = (0, 1, 0, 0.4)))
	# plot the observed t-statictic:
	_ = plt.axvline(x = t_stat, color = "darkgreen", linestyle = "--")
	# Shade the probability alpha / 2
	_ = plt.fill_between(x[x <= -np.abs(t_crit)], hx[x <= -np.abs(t_crit)],
	                 edgecolor = "red", linestyle = "-", linewidth = 2,
	                 facecolor = clrs.to_rgba(c = (1, 0, 0, 0.1)), zorder = 10)
	# Shade the probability 1 - alpha / 2
	_ = plt.fill_between(x[x >= np.abs(t_crit)], hx[x >= np.abs(t_crit)],
	                 edgecolor = "red", linestyle = "-", linewidth = 2,
	                 facecolor = clrs.to_rgba(c = (1, 0, 0, 0.4)), zorder = 10)
	# plot the critical value -|t_c| and t_c|:
	_ = plt.axvline(x = -np.abs(t_crit), color = "red", linestyle = "--")
	_ = plt.axvline(x = np.abs(t_crit), color = "red", linestyle = "--")
	#
	_ = ax.set_xticks([-np.abs(t_crit), t_stat, np.abs(t_crit)])
	_ = ax.set_xticklabels(["$-|t_c|$", "$t_{stat}$", "$|t_c|$"])
	ax.get_xticklabels()[0].set_color("red")
	ax.get_xticklabels()[1].set_color("darkgreen")
	ax.get_xticklabels()[2].set_color("red")
	_ = plt.title("p-value for the two-tail test; $H_1: \\beta_1 \\neq 0$")
	plt.show()

if __name__ == "__main__":
	 plot()