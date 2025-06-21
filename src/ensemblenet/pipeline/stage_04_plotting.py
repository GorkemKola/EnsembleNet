from ensemblenet.config import ConfigurationManager
from ensemblenet.components.plotter import Plotter
from ensemblenet import logger

class PlottingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        plot_config = config.get_plot_config()
        plotter = Plotter(config=plot_config)
        plotter.extract_model_params(save_path=plot_config.root_dir, show_plot=False)
        plotter.plot_test_results(save_path=plot_config.root_dir, columns=['accuracy', 'precision', 'recall'])
        plotter.plot_all_metrics(save_path=plot_config.root_dir / "all_metrics_comparison.png")
        plotter.plot_loss_comparison(save_path=plot_config.root_dir / "loss_comparison.png")
        plotter.plot_accuracy_comparison(save_path=plot_config.root_dir / "accuracy_comparison.png")
        plotter.plot_precision_recall_f1(save_path=plot_config.root_dir / "precision_recall_f1_comparison.png")
        plotter.plot_learning_rate(save_path=plot_config.root_dir / "learning_rate_comparison.png")
        
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
