import unittest
from fastapi.testclient import TestClient
from main import app


class TestPredictLabel(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_predict_label(self):
        with open("Screenshot_1.png", "rb") as image_file:
            response = self.client.post("/net/image/prediction/", files={"image": ("Screenshot_1.png", image_file, "image/jpeg")})
            data = response.json()
            self.assertEqual(response.status_code, 200)
            self.assertIn("model-prediction", data)


if __name__ == '__main__':
    unittest.main()
