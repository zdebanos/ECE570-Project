from schedule_model import ScheduleModel
import sched
from abc import ABC, abstractmethod
from solar_gru import SolarSeq2SeqGRU
from schedule_model.solar_milp_model import SolarMILPModel
from datetime import datetime

# The SolarMPC interface. Defines handy methods.
class SolarMPC(ABC):
    def __init__(
        self,
        forecast_model: SolarSeq2SeqGRU,
        schedule_model: ScheduleModel,
        past_data_minutes=4*24*60,
        time_length=60
    ):
        self.forecast_model = forecast_model
        self.schedule_model = schedule_model

    @abstractmethod
    def _get_input_data(start_date: datetime):
        """Retrieves the input data for the model."""
        pass

    @abstractmethod
    def do_step():
        """Steps the MPC one time step further"""
        pass

