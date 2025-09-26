"""
Centralized logging utility for mBER modules.
Provides consistent logging with both file and console output
and maintains logs for saving to data objects.
"""

import logging
import sys
import io
import threading
import time
from typing import Optional, List, Dict, TextIO, Tuple, Callable, Any
from datetime import datetime


class StdoutCapturer(io.TextIOBase):
    """
    A class to capture stdout and redirect to a logger.
    """
    def __init__(self, log_store, logger, original_stdout):
        self.log_store = log_store
        self.logger = logger
        self.original_stdout = original_stdout
        self.buffer = io.StringIO()
        self.lock = threading.Lock()
        
    def write(self, text):
        # Write to original stdout
        self.original_stdout.write(text)
        
        # Only log non-empty lines
        if text.strip():
            with self.lock:
                # Log to the logger
                self.logger.info(f"[STDOUT] {text.rstrip()}")
                # Store the log
                self.log_store.add_log("info", f"[STDOUT] {text.rstrip()}")
        
        # Write to buffer
        return self.buffer.write(text)
    
    def flush(self):
        self.original_stdout.flush()
        self.buffer.flush()
    
    def isatty(self):
        return self.original_stdout.isatty()


class LogStore:
    """Store and manage logs for later retrieval or saving."""
    
    def __init__(self, name: str):
        """Initialize log store."""
        self.name = name
        self.logs: List[Dict] = []
        self.lock = threading.Lock()
    
    def add_log(self, level: str, message: str):
        """Add a log entry."""
        with self.lock:
            self.logs.append({
                'timestamp': datetime.now().isoformat(),
                'level': level,
                'message': message
            })
    
    def get_logs(self, level: Optional[str] = None) -> List[Dict]:
        """Get logs, optionally filtered by level."""
        with self.lock:
            if level is None:
                return self.logs.copy()
            return [log.copy() for log in self.logs if log['level'] == level]
    
    def get_log_text(self, level: Optional[str] = None) -> str:
        """Get logs as formatted text."""
        logs = self.get_logs(level)
        return "\n".join([
            f"[{log['timestamp']}] {log['level'].upper()}: {log['message']}"
            for log in logs
        ])
    
    def clear(self):
        """Clear all logs."""
        with self.lock:
            self.logs = []


class MberLogger:
    """
    Centralized logging utility for MBER modules.
    Provides methods for logging setup, logging, and teardown.
    """
    
    _loggers = {}  # Cache of created loggers to avoid duplication
    
    @staticmethod
    def setup_logger(
        name: str, 
        verbose: bool = True, 
        log_file: Optional[str] = None, 
        level: int = logging.INFO
    ) -> Tuple[logging.Logger, LogStore]:
        """
        Setup a logger with console and optional file handlers.
        Reuses existing loggers if already set up.
        
        Args:
            name: Name of the logger
            verbose: Whether to print to console in addition to logging
            log_file: Optional path to log file
            level: Logging level (default: INFO)
            
        Returns:
            The configured logger and a log store
        """
        # Check if logger already exists
        if name in MberLogger._loggers:
            return MberLogger._loggers[name]
        
        # Create logger
        logger = logging.getLogger(name)
        
        # Clear any existing handlers for this logger
        if logger.handlers:
            for handler in logger.handlers:
                logger.removeHandler(handler)
        
        # Set logger level
        logger.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Add file handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Create log store
        log_store = LogStore(name)
        
        # Cache the logger and log store
        MberLogger._loggers[name] = (logger, log_store)
        
        return logger, log_store
    
    @staticmethod
    def log(
        logger: logging.Logger, 
        log_store: LogStore, 
        message: str, 
        level: str = "info", 
        verbose: bool = True
    ) -> None:
        """
        Log a message with both logging and optional print statements.
        Also stores the log in the log_store.
        
        Args:
            logger: The logger instance to use
            log_store: The log store to save the message to
            message: Message to log
            level: Log level (info, warning, error, debug)
            verbose: Whether to also print the message
        """
        # Log through the logger
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        elif level == "debug":
            logger.debug(message)
        
        # Store the log
        log_store.add_log(level, message)
        
        # Also print if verbose mode is enabled
        if verbose:
            module_name = logger.name
            print(f"[{module_name}] {message}")
    
    @staticmethod
    def start_stdout_capture(logger: logging.Logger, log_store: LogStore) -> TextIO:
        """
        Start capturing stdout and redirecting to logger.
        
        Args:
            logger: Logger to redirect stdout to
            log_store: Log store to save stdout to
            
        Returns:
            Original stdout to restore later
        """
        original_stdout = sys.stdout
        sys.stdout = StdoutCapturer(log_store, logger, original_stdout)
        return original_stdout
    
    @staticmethod
    def stop_stdout_capture(original_stdout: TextIO) -> None:
        """
        Stop capturing stdout and restore original.
        
        Args:
            original_stdout: Original stdout to restore
        """
        if sys.stdout is not original_stdout:
            sys.stdout = original_stdout
    
    @staticmethod
    def cleanup_logger(name: str) -> None:
        """
        Remove a logger from the cache.
        
        Args:
            name: Name of the logger to remove
        """
        if name in MberLogger._loggers:
            del MberLogger._loggers[name]
    
    @staticmethod
    def cleanup_all() -> None:
        """Clear all cached loggers."""
        MberLogger._loggers.clear()