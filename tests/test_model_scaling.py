import unittest
from models.generate_model_variants import generate_variants


class TestModelScaling(unittest.TestCase):
    def test_generate_variants(self):
        model_name = "VGG11"
        dataset_name = "CIFAR10"
        output_dir = "./models/torchscript_models"
        generate_variants(model_name, dataset_name, output_dir)
        # Add assertions to verify the correctness of the generated variants


if __name__ == "__main__":
    unittest.main()
