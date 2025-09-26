import time
import functools
from contextlib import contextmanager
from typing import Optional, Callable, Dict, Any, List, Union
import logging

# Set up logger
logger = logging.getLogger(__name__)


@contextmanager
def timer(
    name: str,
    logger_func: Optional[Callable] = None,
    store_result: Optional[Dict] = None,
):
    """
    Context manager for timing code blocks.

    Args:
        name: Name of the operation being timed
        logger_func: Function to log the timing result (defaults to print)
        store_result: Optional dictionary to store the timing result

    Yields:
        None

    Example:
        with timer("Load model weights", logger.info):
            model = load_model()
    """
    start_time = time.time()

    try:
        yield
    finally:
        elapsed_time = time.time() - start_time

        # Format timing message
        timing_msg = f"{name} took {elapsed_time:.2f} seconds"

        # Log timing
        if logger_func:
            logger_func(timing_msg)
        else:
            print(timing_msg)

        # Store timing if requested
        if store_result is not None:
            if "timings" not in store_result:
                store_result["timings"] = {}
            store_result["timings"][name] = elapsed_time


def time_method(log_level: str = "info", store_in_state: bool = True):
    """
    Decorator for timing methods, with special handling for MberModule methods.

    Args:
        log_level: Logging level to use ("info", "debug", etc.)
        store_in_state: Whether to store the timing in the design_state object

    Returns:
        Decorated function

    Example:
        @time_method()
        def run(self, design_state):
            # method implementation
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, design_state=None, *args, **kwargs):
            # Get the appropriate logging function
            log_func = None
            if hasattr(self, "_log"):
                log_func = lambda msg: self._log(msg, level=log_level)
            else:
                log_func = print

            method_name = func.__name__
            module_name = self.__class__.__name__
            operation_name = f"{module_name}.{method_name}"

            # Storage for timing results
            timing_storage = {}

            # Time the method execution
            start_time = time.time()
            result = func(self, design_state, *args, **kwargs)
            elapsed_time = time.time() - start_time

            # Log the timing
            timing_msg = f"{operation_name} completed in {elapsed_time:.2f} seconds"
            log_func(timing_msg)

            # Store timing in the design_state if requested
            if store_in_state and design_state is not None:
                # Determine which data object to store timing in
                data_attr = None
                if (
                    method_name == "setup"
                    and hasattr(design_state, "template_data")
                    and design_state.template_data is not None
                ):
                    data_attr = design_state.template_data
                elif method_name == "run":
                    if module_name == "TemplateModule" and hasattr(
                        design_state, "template_data"
                    ):
                        data_attr = design_state.template_data
                    elif module_name == "TrajectoryModule" and hasattr(
                        design_state, "trajectory_data"
                    ):
                        data_attr = design_state.trajectory_data
                    elif module_name == "EvaluationModule" and hasattr(
                        design_state, "evaluation_data"
                    ):
                        data_attr = design_state.evaluation_data
                elif method_name == "teardown" and hasattr(
                    design_state, f"{module_name.lower().replace('module', '')}_data"
                ):
                    data_attr = getattr(
                        design_state,
                        f"{module_name.lower().replace('module', '')}_data",
                    )

                # Store timing data
                if data_attr is not None:
                    if not hasattr(data_attr, "timings"):
                        data_attr.timings = {}
                    data_attr.timings[method_name] = elapsed_time

            return result

        return wrapper

    return decorator


class TimingBlock:
    """
    A reusable class for timing blocks of code with explicit start/stop.

    Example:
        timer = TimingBlock("Operation name")
        timer.start()
        # ... do something ...
        timer.stop()
        print(timer.elapsed)
    """

    def __init__(
        self,
        name: str,
        logger_func: Optional[Callable] = None,
        auto_start: bool = False,
        store_result: Optional[Dict] = None,
    ):
        self.name = name
        self.logger_func = logger_func
        self.store_result = store_result
        self.start_time = None
        self.elapsed = None

        if auto_start:
            self.start()

    def start(self):
        """Start the timer"""
        self.start_time = time.time()
        return self

    def stop(self):
        """Stop the timer and calculate elapsed time"""
        if self.start_time is None:
            raise ValueError("Timer was not started")

        self.elapsed = time.time() - self.start_time

        # Format timing message
        timing_msg = f"{self.name} took {self.elapsed:.2f} seconds"

        # Log timing
        if self.logger_func:
            self.logger_func(timing_msg)
        else:
            print(timing_msg)

        # Store timing if requested
        if self.store_result is not None:
            if "timings" not in self.store_result:
                self.store_result["timings"] = {}
            self.store_result["timings"][self.name] = self.elapsed

        return self.elapsed

    def __enter__(self):
        """Support context manager interface"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support context manager interface"""
        self.stop()
        # Don't suppress exceptions
        return False
