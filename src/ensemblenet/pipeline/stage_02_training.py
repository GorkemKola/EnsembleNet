from ensemblenet.config import ConfigurationManager
from ensemblenet.components.training import Training
from ensemblenet import logger
import dataclasses

class TrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.prepare_data_loaders()
        training.build_model()
        
        if training_config.params_find_lr:
            lr = training.find_optimal_lr_ignite()
            training.config = dataclasses.replace(training.config, params_learning_rate=lr)

        training.train()
        
STAGE_NAME = 'Training Stage'

if __name__ == '__main__':
    try:
        logger.info(
            f'>>>>> stage {STAGE_NAME} started <<<<<'
        )
        obj = TrainingPipeline()
        obj.main()
        logger.info(
              f'>>>>> stage {STAGE_NAME} completed <<<<<\n\nx===========x'
        )

    except Exception as e:
        logger.info(e)
        raise e
