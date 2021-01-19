import pickle
import lightgbm as lgb

def predict(query):
	"""
	@query: Input vector
	Returns: Ranking score for the relavance of the query (measuring the compatibility of job and student)
	"""
	
	if len(query.shape) == 1:
		query = query[np.newaxis, :]
	model = None
	with open("samples/model.p", "rb") as f:
		model = pickle.load(f)
	return model.predict(query)