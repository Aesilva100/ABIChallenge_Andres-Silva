import unittest, joblib, os

class TestModeloIris(unittest.TestCase):

    def setUp(self):
        # Here, the model is set as an instance variable using 'self.model'
        model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'gaussian_nb_model.pkl')
        self.model = joblib.load(model_path)

    def test_prediccion_individual(self):
        # 'self.model' is used to access the loaded model
        test_data = [5.1, 3.5, 1.4, 0.2]
        result = self.model.predict([test_data])  # 'model' is changed to 'self.model'
        self.assertEqual(len(result), 1)  # Verifies that the result is a single prediction

    def test_prediccion_por_lote(self):
        # 'self.model' is used to access the loaded model
        test_data = [
            [5.1, 3.5, 1.4, 0.2],
            [6.2, 3.4, 5.4, 2.3]
        ]
        results = self.model.predict(test_data)  # 'model' is changed to 'self.model'
        self.assertEqual(len(results), 2)  # Verifies that there is one prediction for each set of features

if __name__ == '__main__':
    unittest.main()
