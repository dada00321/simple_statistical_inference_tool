""" Randomized Block Design """

def get_y_bar(data):
	c1, c2 = get_constants(data)
	y_bar = sum([e for row in data for e in row])/(c1*c2)
	return y_bar

def get_constants(data):
	c1 = len(data)
	c2 = len(data[0])
	return c1, c2

def get_y_dot_j_list(data):
	c1, c2 = get_constants(data)
	y_dot_j_list = [sum([row[j] for row in data])/c1 for j in range(c2)]
	return y_dot_j_list
	
def get_y_i_dot_list(data):
	c1, c2 = get_constants(data)
	y_i_dot_list = [sum(row)/c2 for row in data]
	return y_i_dot_list
	
# Factor Sum of Squares
def FSS(data):
	# ΣΣ(Y.j - Ȳ)^2
	c1, c2 = get_constants(data)
	y_bar = get_y_bar(data)
	y_dot_j_list = get_y_dot_j_list(data)
	#-----------------------------
	fss = sum([c1*(y_dot_j - y_bar)**2 for y_dot_j in y_dot_j_list])
	return fss

# Block Sum of Squares
def BSS(data):
	# ΣΣ(Yi. - Ȳ)^2
	c1, c2 = get_constants(data)
	y_bar = get_y_bar(data)
	y_i_dot_list = get_y_i_dot_list(data)
	#-----------------------------
	bss = sum([c2*(y_i_dot - y_bar)**2 for y_i_dot in y_i_dot_list])
	return bss

def TSS(data):
	# ΣΣ(Yij - Ȳ)^2
	y_bar = get_y_bar(data)
	#-----------------------------
	tss = sum([(y_i_j - y_bar)**2 for row in data for y_i_j in row])
	return tss
	
# Error Sum of Squares
def ESS(data):
	# ΣΣ(Yij -Yi. -Y.j + Ȳ)^2
	# ESS also e.q. to "TSS-BSS-FSS"
	# c1, c2 = get_constants(data)
	# y_bar = get_y_bar(data)
	# y_i_dot_list = get_y_i_dot_list(data)
	# y_dot_j_list = get_y_dot_j_list(data)
	#-----------------------------
	ess = TSS(data) - BSS(data) - FSS(data)
	return ess

# List of "Sum of Squares" of source (source: Factor/Block/Error)
def SS_list(data):
	return {"FSS": convert_precision(FSS(data)),
	        "BSS": convert_precision(BSS(data)),
			"ESS": convert_precision(ESS(data)),
			"TSS": convert_precision(TSS(data))}

# List of "Degree of Freedom"
def DF_list(data):
	c1, c2 = get_constants(data)
	return {"ν(Factor)": c2-1,
	        "ν(Block)": c1-1,
			"ν(Error)": (c1-1)*(c2-1),
			"ν(Total)": c1*c2-1}

# List of "Mean Sum" of source (source: Factor/Block/Error)
# MS = SS / DF
def MS_list(data):
	ss_list = SS_list(data)
	df_list = DF_list(data)
	return {"MSF": 
			convert_precision(ss_list["FSS"] / df_list["ν(Factor)"]),
	        "MSB": 
			convert_precision(ss_list["BSS"] / df_list["ν(Block)"]),
			"MSE": 
			convert_precision(ss_list["ESS"] / df_list["ν(Error)"])}

# List of "F value" (i.e. F1 and F2)
def F_list(data):
	ms_list = MS_list(data)
	return {"F1": convert_precision(ms_list["MSF"] / ms_list["MSE"]),
	        "F2": convert_precision(ms_list["MSB"] / ms_list["MSE"])}

# Convert precision for floating point number
def convert_precision(number, precision=4):
	return round(number, precision)

# Print out ANOVA Table simply!
def show_ANOVA_Table(data):
	print('='*15, "  ANOVA Table", '='*15, sep='\n')
	print("  ", *(f"{text}\t\t" for text in ("Source", "SS", "DF", "MS", "F")), sep='')
	ss_list = SS_list(data)
	df_list = DF_list(data)
	ms_list = MS_list(data)
	f_list = F_list(data)
	# (1) Factor
	print("  Factor\t"+\
		  f'  FSS={ss_list["FSS"]}\t\t{df_list["ν(Factor)"]}\t\t'+\
	      f'{ms_list["MSF"]}\t\t{f_list["F1"]}')
	# (2) Block
	print("  Block\t\t"+\
		  f'  BSS={ss_list["BSS"]}\t\t{df_list["ν(Block)"]}\t\t'+\
	      f'{ms_list["MSB"]}\t\t{f_list["F2"]}')
	# (3) Error
	print("  Error\t\t"+\
		  f'  ESS={ss_list["ESS"]}\t\t{df_list["ν(Error)"]}\t\t'+\
	      f'{ms_list["MSE"]}')
	# (4) Total
	print("  Total\t\t"+\
		  f'  TSS={ss_list["TSS"]}\t\t{df_list["ν(Total)"]}\t\t')

if __name__ == "__main__":
	"""
	data = [[10,12,18],
			[11,14,8],
			[10,12,8],
			[9,10,8],
			[10,12,8]]
	"""
	data = [[13,12,10],
			[10,13,9],
			[14,18,10],
			[8,11,7]]
	
	'''
	print(FSS(data))
	print(BSS(data))
	print(ESS(data))
	print(TSS(data))
	'''
	
	'''
	print(SS_list(data))
	print(DF_list(data))
	print(MS_list(data))
	print(F_list(data))
	'''
	
	show_ANOVA_Table(data)