from abc import ABC, abstractmethod

class ScheduleModelParameters(ABC):
    """Simple Schedule Model Data Holder for variables"""
    pass

class ScheduleModel(ABC):
    """Simple Schedule Model Interface"""
    def __init__(self, time_steps: int, dt: float, params: ScheduleModelParameters):
        """Initializes the Schedule Model

        Args:
            self
            time_steps: number of time steps
            dt:         a length of a time step in minutes
            parameters: a ScheduleModelParameters object
        """
        self.params = params
        self.time_steps = time_steps
        self.dt = dt

    @abstractmethod
    def solve(self):
        pass
