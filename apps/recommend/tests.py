from django.test import TestCase, SimpleTestCase
import pickle
import numpy as np
from apps.recommend.rec_utils import *
from apps.recommend.rec_model import *

embeddings = None
with open("apps/recommend/embeddings.p", "rb") as f:
	embeddings = pickle.load(f)

class TestRecUtils(SimpleTestCase):
	def test_doc2vec_shape(self):
		inp_text = "Hello World"
		doc2vec_output = doc2vec(inp_text, embeddings)

		assert doc2vec_output.shape == (1, 50)

	def test_doc2vec_simple_input(self):
		inp_text = "Hello World"
		expected_output = (embeddings["hello"] + embeddings["world"])/2
		doc2vec_output = doc2vec(inp_text, embeddings)

		assert np.isclose(doc2vec_output, expected_output).all()

	def test_doc2vec_word_order(self):
		inp_text_1, inp_text_2 = "Hello World", "World Hello"
		doc2vec_output_1 = doc2vec(inp_text_1, embeddings)
		doc2vec_output_2 = doc2vec(inp_text_2, embeddings)

		assert np.isclose(doc2vec_output_1, doc2vec_output_2).all()

