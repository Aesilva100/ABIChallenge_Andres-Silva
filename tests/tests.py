import unittest, joblib, os

class TestModeloIris(unittest.TestCase):

    def setUp(self):
        # Aquí, se establece el modelo como una variable de instancia usando 'self.model'
        model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'gaussian_nb_model.pkl')
        self.model = joblib.load(model_path)

    def test_prediccion_individual(self):
        # Se utiliza 'self.model' para acceder al modelo cargado
        datos_de_prueba = [5.1, 3.5, 1.4, 0.2]
        resultado = self.model.predict([datos_de_prueba])  # Se cambia 'model' por 'self.model'
        self.assertEqual(len(resultado), 1)  # Verifica que el resultado sea una sola predicción

    def test_prediccion_por_lote(self):
        # Se utiliza 'self.model' para acceder al modelo cargado
        datos_de_prueba = [
            [5.1, 3.5, 1.4, 0.2],
            [6.2, 3.4, 5.4, 2.3]
        ]
        resultados = self.model.predict(datos_de_prueba)  # Se cambia 'model' por 'self.model'
        self.assertEqual(len(resultados), 2)  # Verifica que haya una predicción por cada conjunto de características

if __name__ == '__main__':
    unittest.main()
