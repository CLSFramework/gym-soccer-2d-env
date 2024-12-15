from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import logging

class InfoCollectorCallback(BaseCallback):
    """
    Callback for collecting information from the environment during training.
    """
    def __init__(self):
        """
        Initialize the InfoCollectorCallback.
        """
        super().__init__()
        self.infos = []  # To store the info dictionaries
        self.results = {}

    def _on_step(self) -> bool:
        """
        Collect the info dictionary from the environment at each step.
        
        :return: True to continue training.
        """
        if self.locals.get('infos') is not None:
            infos = self.locals.get('infos')
            for info in infos:
                if info['result'] and len(info['result']) > 0:
                    self.infos.append(info)
        return True  # Continue training
    
    def reset(self):
        """
        Reset the collected information.
        """
        self.infos = []
        self.results = {}
        
    def update_results_dict(self, logger: logging.Logger) -> dict:
        """
        Update the results dictionary with the collected information.
        
        :param logger: Logger instance for logging.
        :return: Dictionary of results.
        """
        results_types = ['Goal', 'Out', 'Timeout']
        results = [info['result'] for info in self.infos]
        self.results = {type: [] for type in results_types}
        
        for i in range(0, len(results), 100):
            for type in results_types:
                length = len(results[i:i+100])
                self.results[type].append(results[i:i+100].count(type) / length)
        
        logger.info(f"Results dictionary: {self.results}")
    
    def plot_print_results(self, logger: logging.Logger, file_name: str = None) -> tuple:
        """
        Plot and print the results from the collected information.
        
        :param logger: Logger instance for logging.
        :param file_name: Optional file name to save the plot.
        :return: Tuple containing lists of results for 'Goal', 'Out', and 'Timeout'.
        """
        self.update_results_dict(logger)
        
        fig, ax = plt.subplots()
        for type in self.results:
            ax.plot(self.results[type], label=type)
            
        ax.legend()
        ax.set_xlabel('Episodes (x100)')
        ax.set_ylabel('Percentage')
        ax.set_title('Results')
        
        if file_name:
            plt.savefig(file_name + '.png')
            logger.info(f"Plot saved as {file_name}.png")
        else:
            plt.show()
            logger.info("Plot displayed.")
        
        return self.results['Goal'], self.results['Out'], self.results['Timeout']