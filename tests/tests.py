import unittest, joblib
class TestModeloIris(unittest.TestCase):
    model=joblib.load("..model/gaussian_nb_model.pkl")
    def test_prediccion_individual(self):
        # Supongamos que tu función de predicción espera una lista de características
        # Cambia los valores de prueba según tu modelo específico
        datos_de_prueba = [5.1, 3.5, 1.4, 0.2]
        resultado = model([datos_de_prueba])  # Envía los datos como un lote de un solo elemento
        self.assertEqual(len(resultado), 1)  # Asegura que el resultado sea una sola predicción

    def test_prediccion_por_lote(self):
        # Supongamos que tu función de predicción puede manejar varios conjuntos de características
        datos_de_prueba = [
            [5.1, 3.5, 1.4, 0.2],
            [6.2, 3.4, 5.4, 2.3]
        ]
        resultados = model(datos_de_prueba)
        self.assertEqual(len(resultados), 2)  # Asegura que haya una predicción por cada conjunto de características

if __name__ == '__main__':
    unittest.main()
