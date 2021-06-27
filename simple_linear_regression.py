""" Simple Linear Regression """

def get_x_bar(data):
	x_bar = sum(data[0])/len(data[0])
	return x_bar

def get_y_bar(data):
	y_bar = sum(data[1])/len(data[1])
	return y_bar

def get_b_1(data):
	x, y = data[0], data[1]
	x_bar = get_x_bar(data)
	b_1_dividend = sum([(x_i-x_bar) * y_i for x_i, y_i in zip(x,y)])
	b_1_divisor = sum([(x_i-x_bar)**2 for x_i, y_i in zip(x,y)])
	return b_1_dividend / b_1_divisor

def get_b_0(data):
	x_bar = get_x_bar(data)
	y_bar = get_y_bar(data)
	b_1 = get_b_1(data)
	b_0 = y_bar - b_1 * x_bar
	return b_0

def get_y_i_head(x_i, b_0, b_1):
	y_i_head = b_0 + b_1 * x_i
	return y_i_head

# n: number of data (data: X / Y)
def get_n(data):
	return len(data[0])

# s: stardard error
def get_s(mse):
	return mse**0.5

def get_r_square(data, y_bar, ess):
	'''
	r_square = sum([(get_y_i_head(x_i, b_0, b_1) - y_bar)**2 
					for x_i, y_i in zip(data[0],data[1])]) / 
			   sum([(y_i - y_bar)**2 
					for x_i, y_i in zip(data[0],data[1])])
	'''
	tss = sum([(y_i - y_bar)**2 
				for x_i, y_i in zip(data[0],data[1])])
	r_square = (tss-ess) / tss		
	return r_square

def ESS(data):
	b_0, b_1 = get_b_0(data), get_b_1(data)
	'''
	ess = 0
	for x_i, y_i in zip(data[0], data[1]):
		y_i_head = get_y_i_head(x_i, b_0, b_1)
		ess += (y_i - y_i_head)**2
	'''
	ess = sum([(y_i - get_y_i_head(x_i, b_0, b_1))**2
				for x_i, y_i in zip(data[0], data[1])])
	return ess

def MSE(ess, n):
	return ess / (n-2)

def get_t_observed(data, b_1, baseline, s):
	x_bar = get_x_bar(data)
	t_observed = (b_1 - baseline) / (s/ sum([(x_i - x_bar)**2 for x_i in data[0]])**0.5)
	return t_observed

def predict_CI_for_y_bar(data, x_0_for_y_bar, b_0, b_1, n, s, test_statistics_t_value):
	x_bar = get_x_bar(data)
	y_0_head = get_y_i_head(x_0_for_y_bar, b_0, b_1)
	variable_range = test_statistics_t_value * s * ( 1/n + (x_0_for_y_bar - x_bar)**2 / sum([(x_i - x_bar)**2 for x_i in data[0]]) )**0.5
	print(f"ŷ_0 = {y_0_head}")
	print(f"variable_range = {convert_precision(variable_range)}")
	return (convert_precision(y_0_head-variable_range), 
			convert_precision(y_0_head+variable_range))

def predict_CI_for_y_i(data, x_0_for_y_i, b_0, b_1, n, s, test_statistics_t_value):
	x_bar = get_x_bar(data)
	y_0_head = get_y_i_head(x_0_for_y_bar, b_0, b_1)
	variable_range = test_statistics_t_value * s * ( 1 + 1/n + (x_0_for_y_bar - x_bar)**2 / sum([(x_i - x_bar)**2 for x_i in data[0]]) )**0.5
	print(f"ŷ_0 = {y_0_head}")
	print(f"variable_range = {convert_precision(variable_range)}")
	return (convert_precision(y_0_head-variable_range), 
			convert_precision(y_0_head+variable_range))

# Convert precision for floating point number
def convert_precision(number, precision=4):
	return round(number, precision)

# Print out inference of Simple Linear Regression simply!
def show_inference(data, baseline, alpha, CI_range, test_statistics_t_value, x_0_for_y_bar, x_0_for_y_i):
	print('='*30, "    簡單線性迴歸的推論統計", '='*30, sep='\n')
	
	print("Step 1. 計算迴歸方程式（此為直線方程式）的迴歸係數，並找出迴歸方程式")
	b_0 = get_b_0(data)
	b_1 = get_b_1(data)
	print(f"\nb_0 = {b_0}  // i.e. 截距（intercept）")
	print(f"b_1 = {b_1}  // i.e. 斜率（slope）")
	print(f"\nRegression Equation: ŷ_i = {b_0} + {b_1} * X_i")
	
	print('-'*30, '\n', "Step 2. 計算判定係數（R^2, coefficient of determination）", sep='')
	y_bar = get_y_bar(data)
	ess = ESS(data)
	r_square = get_r_square(data, y_bar, ess)
	print(f"R^2 = RSS/TSS = {r_square}")
	
	print('-'*30, '\n', "Step 3. 對迴歸係數 β_1 進行假設檢定", sep='')
	
	print("\n(1) 虛無假設 與 對立假設")
	print(f"H_0: β_1 ≦ {baseline}")
	print(f"H_1: β_1 ＞ {baseline}")
	
	print("\n(2) 顯著水準 與 樣本數")
	n = get_n(data)
	print(f"α = {alpha}, n = {n}")
	
	print("\n(3) 檢定統計量 與 虛無分配")
	print(f"(b_1 - β_1) / (s / sqrt(Σ(X_i - X̄)^2)) ~ t(n-2) = t({n-2})")
	print(f"// 需查表後給定 t({n-2}) 數值 ~> `test_statistics_t_value`: {test_statistics_t_value}")
	
	print("\n(4) 決策法則")
	print(f"(畫出決策法則)")
	
	print("\n(5) 先求算 s (標準誤)")
	print(f"● 用 MSE 求算 s^2 (=> 估計 σ^2)")
	
	mse = MSE(ess, n)
	s = get_s(mse)
	
	print(f"ESS = Σ(Y_i - ŷ_i)^2 = {ess}")
	print(f"MSE = ESS / (n-2) = s^2 = {mse}")
	print(f"s = sqrt(MSE) = {s}")
	
	print("\n(6) 推論")
	t_observed = get_t_observed(data, b_1, baseline, s)
	print(f"t_(observed) = (b_1 - β_1) / (s / sqrt(Σ(X_i - x̄)^2)) = {convert_precision(t_observed)}")
	
	# <右尾>
	if t_observed > test_statistics_t_value: 
		# reject H_0, believe β_1 > baseline
		print(f"∵ t_(observed) = {t_observed} > t(n-2)=t({n-2})={test_statistics_t_value}")
		print(f"∴ Reject H_0, believe β_1 > {baseline}")
	else:
		# do not (fail to) reject H_0, believe β_1 <= baseline
		print(f"∵ t_(observed) = {t_observed} ≦ t(n-2)=t({n-2})={test_statistics_t_value}")
		print(f"∴ Do not (fail to) reject H_0, believe β_1 <= {baseline}")

	print('-'*30, '\n', f"Step 4. 求出 預測平均數(Ȳ) 的 {CI_range}% 信賴區間（given X_0 = {x_0_for_y_bar}）", sep='')
	ci_for_y_bar = predict_CI_for_y_bar(data, x_0_for_y_bar, b_0, b_1, n, s, test_statistics_t_value)
	print("\n信賴區間:")
	print(" ŷ_0 ± t(n-2) * s * sqrt((1/n)+((X_0 - X̄)^2 / Σ_i(X_i - X̄)^2))")
	print(f" = {ci_for_y_bar}")
	
	print('-'*30, '\n', f"Step 5. 求出 個別觀測值(Y_i) 的 {CI_range}% 預測區間（given X_0 = {x_0_for_y_i}）", sep='')
	ci_for_y_i = predict_CI_for_y_i(data, x_0_for_y_i, b_0, b_1, n, s, test_statistics_t_value)
	print("\n預測區間:")
	print(" ŷ_0 ± t(n-2) * s * sqrt(1+(1/n)+((X_0 - X̄)^2 / Σ_i(X_i - X̄)^2))")
	print(f" = {ci_for_y_i}")
	
if __name__ == "__main__":
	'''
	data = [[2,6,4,2,8,8,6,4],
			[24,28,24,20,30,26,24,24]]
	'''
	data = [[6,3,2,9,5,7,2,8],
			[31,56,48,84,36,73,17,94]]
	baseline = 0
	alpha = 0.05 # significance level
	CI_range = 95
	x_0_for_y_bar = 3
	x_0_for_y_i = 3
	'''
	print(get_b_1(data))
	print(get_b_0(data))
	'''
	#test_statistics_t_value = 1.943 # α=0.05下, t(8-2)=t(6)=1.943 (<單尾>)
	test_statistics_t_value = 1.9432 # α=0.05下, t(8-2)=t(6)=1.943 (<單尾>)
	#test_statistics_t_value = 2.4469 # α=0.05下, t(8-2)=t(6)=1.943 (<單尾>)
	show_inference(data, baseline, alpha, CI_range, test_statistics_t_value, x_0_for_y_bar, x_0_for_y_i)
	