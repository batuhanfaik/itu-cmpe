import pickle
import numpy as np

from django.test import TestCase, SimpleTestCase
from apps.recommend.rec_utils import *
from apps.recommend.rec_model import *
from apps.job.models import Job
from apps.student.models import Student

class TestRecUtils(SimpleTestCase):

	def doc2vec_setup(self):
		self.embeddings = None
		with open("apps/recommend/embeddings.p", "rb") as f:
			self.embeddings = pickle.load(f)

	def create_features_setup(self):
		pass

	def test_doc2vec_shape(self):
		inp_text = "Hello World"
		doc2vec_output = doc2vec(inp_text, self.embeddings)

		assert doc2vec_output.shape == (1, 50)

	def test_doc2vec_simple_input(self):
		inp_text = "Hello World"
		expected_output = (self.embeddings["hello"] + self.embeddings["world"])/2
		doc2vec_output = doc2vec(inp_text, self.embeddings)

		assert np.isclose(doc2vec_output, expected_output).all()

	def test_doc2vec_word_order(self):
		inp_text_1, inp_text_2 = "Hello World", "World Hello"
		doc2vec_output_1 = doc2vec(inp_text_1, self.embeddings)
		doc2vec_output_2 = doc2vec(inp_text_2, self.embeddings)

		assert np.isclose(doc2vec_output_1, doc2vec_output_2).all()

	def test_create_features_available_skills(self):
		pass

