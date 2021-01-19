import pickle
import lightgbm as lgb
import numpy as np

def predict(query):
	"""
	@query: Input vector
	Returns: Ranking score for the relavance of the query (measuring the compatibility of job and student)
	"""
	
	if len(query.shape) == 1:
		query = query[np.newaxis, :]
	model = None
	with open("apps/recommend/samples/model.p", "rb") as f:
		model = pickle.load(f)

	retval = model.predict(query)
	retval = retval * 25
	retval = retval + 50
	retval = retval[0]
	if(retval > 100):
		retval = 100
	if(retval < 0):
		retval = 0

	retval = round(retval)

	return retval