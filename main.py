from ensemblenet import logger
from ensemblenet.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from ensemblenet.pipeline.stage_02_training import TrainingPipeline
from ensemblenet.pipeline.stage_03_testing import TestingPipeline
from ensemblenet.pipeline.stage_04_plotting import PlottingPipeline



STAGE_NAME = 'Plotting Stage'

if __name__ == '__main__':
    try:
        logger.info(
            f'>>>>> stage {STAGE_NAME} started <<<<<'
        )
        obj = PlottingPipeline()
        obj.main()
        logger.info(
              f'>>>>> stage {STAGE_NAME} completed <<<<<\n\nx===========x'
        )

    except Exception as e:
        logger.info(e)
        raise e