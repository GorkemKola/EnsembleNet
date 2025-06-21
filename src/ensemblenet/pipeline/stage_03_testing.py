from ensemblenet.config import ConfigurationManager
from ensemblenet.components.test import Test
from ensemblenet import logger

class TestingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        test_config = config.get_test_config()
        test = Test(config=test_config)
        test.load_models()
        test.evaluate_all_models()
        
STAGE_NAME = 'Evaluation Stage'

if __name__ == '__main__':
    try:
        logger.info(
            f'>>>>> stage {STAGE_NAME} started <<<<<'
        )
        obj = TestingPipeline()
        obj.main()
        logger.info(
              f'>>>>> stage {STAGE_NAME} completed <<<<<\n\nx===========x'
        )

    except Exception as e:
        logger.info(e)
        raise e
