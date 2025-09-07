# server.py
from ultralytics import YOLO
from collections import OrderedDict

from strategy import Strategy

class Server:
    """
    Server base class
    """
    def __init__(self, strategy:Strategy, yolo_config:str, communication_rounds:int=1):
        self.strategy = strategy
        self.communication_rounds = communication_rounds

        # global model (server owns this)
        self.global_model = YOLO(yolo_config)
        # state dict of global model
        self.global_state = OrderedDict({k : v.clone().cpu() for k, v in self.global_model.state_dict().items()})
    
    def update_global_state(self, new_state:OrderedDict):
        """
        updates the global state variable to the new state

        Args:
            new_state (OrderedDict): New parameters statedict
        """
        self.global_state = OrderedDict({k: v.clone().cpu() for k, v in new_state.items()})
        # update in-memory model too
        self.global_model.load_state_dict(self.global_state)
