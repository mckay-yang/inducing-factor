import logging

class BaseLoader:
    """
    Base data loader class, used to unify the reading interface of various data sources.
    """
    def __init__(self, file_path: str = None, config: dict = None):
        self.file_path = file_path
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        # If you need to set the format or level of each logger separately, you can add it here

    def load(self):
        """
        Execute specific data reading logic, return the processed data structure (such as array, DataFrame or xarray).
        """
        raise NotImplementedError("Subclasses must implement the load method")

    def check_data(self):
        """
        Used for preliminary data checking (such as dimensions, NoData values, ranges, etc.), commonly used for debugging and logging.
        """
        raise NotImplementedError("Subclasses must implement the check_data method")
