import numpy as np

class Preprocess_LC:

	def __init__(self, data, mjd, error):

		self.N = len(mjd)
		self.m = np.mean(error)
		self.mjd = mjd
		self.data = data
		self.error = error

	def Preprocess(self):

		mjd_out = []
		data_out = []
		error_out = []	

		for i in xrange(len(self.data)):
					
					if self.error[i] < (3 * self.m) and (np.absolute(self.data[i] - np.mean(self.data)) / np.std(self.data)) < 5 :
						
						mjd_out.append(self.mjd[i])
						data_out.append(self.data[i])
						error_out.append(self.error[i])


		data_out = np.asarray(data_out)	
		mjd_out = np.asarray(mjd_out)	
		error_out = np.asarray(error_out)	

		return [data_out, mjd_out, error_out]
