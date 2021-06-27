""" Chi square test --(1) Goodness-of-fit test  """
from scipy.stats import chi2
import numpy as np
import matplotlib.pyplot as plt

class Goodness_of_fit_test():
	def __init__(self,
			     sample_size,
				 population_proportions,
				 sample_occurences,
				 significance_level):
		 self.is_available = True
		 if sample_size >= 30:
			  self.sample_size = sample_size
		 else:
			  print("[WARNING] 卡方檢定 基於大樣本假設\n"+\
					  "若樣本數不足 30，樣本仍服從二項分配，\n但不能推導出: 檢定統計量 ==近似於==> 卡方分配\n"+\
					  "i.e. If N < 30\n	  then  n_i ~ Binomial(N, π_i)\n"+\
					  "	  and  \"(n_i-N*π_i) / sqrt(N*π_i) → Z\" is false\n"+\
					  "	  Thus, \"(n_i-N*π_i)^2 / (N*π_i) → χ2\" is false\n")
			  self.is_available = False

		 if abs(sum(population_proportions)-1) < 0.0001:
			 self.population_proportions = population_proportions
		 else:
			 print("[WARNING] Input is invalid. The sum of population proportions is not equal to 1 obviously.")
			 self.is_available = False

		 if 0.001 <= significance_level <= 0.995:
			 self.significance_level = significance_level
		 else:
			 print("[WARNING] Significance level your given is extreme.")
			 self.is_available = False

		 if self.is_available:
			  self.sample_occurences = sample_occurences
			  self.degree_of_freedom = len(population_proportions)-1

	def get_test_statistic(self):
		if self.is_available:
			tmp = list()
			for i, population_proportion in enumerate(self.population_proportions):
				e_i = self.sample_size*population_proportion
				o_i = self.sample_occurences[i]
				tmp.append((o_i-e_i)**2/e_i)
			return sum(tmp)

	def get_chi_square(self):
		# 即"查表"的步驟，利用 scipy 套件完成
		if self.is_available:
			return chi2.ppf(1-self.significance_level, self.degree_of_freedom)

	def plot_null_distribution(self, comparison_msg, observed_test_statistic):
		fig, ax = plt.subplots(1, 1)
		mean, var, skew, kurt = chi2.stats(self.degree_of_freedom, moments='mvsk')

		x = np.linspace(chi2.ppf(0.001, self.degree_of_freedom),
				        chi2.ppf(0.999, self.degree_of_freedom), self.sample_size)
		ax.plot(x, chi2.pdf(x, self.degree_of_freedom),
       'r-', lw=5, alpha=1-self.significance_level, label='chi2 pdf')

		rv = chi2(self.degree_of_freedom)
		ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

		r = chi2.rvs(self.degree_of_freedom, size=self.sample_size)
		ax.hist(r, density=True, histtype='stepfilled', alpha=1-self.significance_level)
		ax.legend(loc='best', frameon=False)
		plt.text(observed_test_statistic, 0.3, comparison_msg)
		plt.show()

	def show_solution(self):
		  '''
		  print("H_0:",
				*(f" π_{i} =" for i in range(1, 1+len(population_proportions))),
				sep='')
		  '''
		  if self.is_available:
			  H_0 = ["H_0: 母體的真實分配是 (", ")"]
			  H_0.insert(1, ", ".join([f"{round(e, 4)}" for e in self.population_proportions]))
			  H_0 = ''.join(H_0)
			  self.H_0 = H_0
			  self.H_1 = H_0.replace('是', "並非")

			  print("Step 1. 設立假設")
			  print("  ", self.H_0)
			  print("  ", self.H_1)

			  print("\nStep 2.")
			  print(f"   α = {self.significance_level}, "+\
			        f"N = {self.sample_size}")

			  print("\nStep 3. 決定 檢定統計量 及 虛無分配")
			  print(f"Σ_i=1, k={self.degree_of_freedom+1} (n_i-N*π_i)^2 / (N*π_i) → χ2({self.degree_of_freedom})")
			  expected_chi_square = self.get_chi_square()
			  print(f"其中 χ2({self.degree_of_freedom}) = {expected_chi_square}")

			  print("\nStep 4. 決策法則")
			  print("(畫出決策法則)")

			  print("\nStep 5. 計算觀察樣本的 檢定統計量")
			  print(f" Σ_i=1, k={self.degree_of_freedom+1} (o_i-e_i)^2 / e_i",
				    f" = Σ_i=1, k={self.degree_of_freedom+1} (n_i-N*π_i)^2 / (N*π_i)",
					sep='\n', end="\n = ")
			  print(*(f"({round(self.sample_occurences[i], 4)} - {round(self.sample_size*population_proportion, 4)})^2/{round(self.sample_size*population_proportion, 4)}"
				    for i, population_proportion in enumerate(self.population_proportions)),
				    sep=" +\n   ", end="\n = ")
			  observed_test_statistic = self.get_test_statistic()

			  '''
			  print(observed_test_statistic)
			  if observed_test_statistic > expected_chi_square:
				  print(f" ＞ χ2({self.degree_of_freedom}) = {expected_chi_square}")
			  else:
				  print(f" ≦ χ2({self.degree_of_freedom}) = {expected_chi_square}")
			  '''
			  comparison_msg = ""
			  comparison_msg += str(observed_test_statistic) + '\n'
			  if observed_test_statistic > expected_chi_square:
				  comparison_msg += f" ＞ χ2({self.degree_of_freedom}) = {expected_chi_square}"
			  else:
				  comparison_msg += f" ≦ χ2({self.degree_of_freedom}) = {expected_chi_square}"
			  print(comparison_msg)

			  print("\nStep 6. 推論")
			  # <右尾>
			  if observed_test_statistic > expected_chi_square:
				  print("∴ Reject H_0,"+\
						f"\n   相信 {self.H_1}")
			  else:
				  print("∴ Do not (Fail to) reject H_0,"+\
						f"\n   相信 {self.H_0}")
			  self.plot_null_distribution(comparison_msg, observed_test_statistic)

if __name__ == "__main__":
	 sample_size = 150
	 population_proportions = [1/3]*3
	 sample_occurences = [60, 50, 40]
	 alpha = 0.01
	 goft = Goodness_of_fit_test(sample_size, population_proportions, sample_occurences, alpha)
	 goft.show_solution()
	 #goft.plot_null_distribution()
