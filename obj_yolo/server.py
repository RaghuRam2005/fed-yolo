#server.py
from typing import List

from ultralytics.engine.model import Model

from .strategy import Strategy
from .client import Client, weather_tags, scene_tags

class Server:
    def __init__(self, communication_rounds:int, model:Model, strategy:Strategy, num_nodes:int, experiment_type: str):
        self.communication_rounds = communication_rounds
        self.global_model = model
        self.global_state = self.global_model.model.model.state_dict()
        self.strategy = strategy
        self.num_nodes = num_nodes
        self.experiment_type = experiment_type
    
    def create_clients(self) -> List[Client]:
        clients = []
        if self.experiment_type == "weather":
            tags = weather_tags
        elif self.experiment_type == "scene":
            tags = scene_tags
        else:
            raise ValueError("Invalid experiment_type. Choose 'weather' or 'scene'.")

        for num in range(self.num_nodes):
            # Cycle through tags if there are more clients than available tags
            client_tag = tags[num % len(tags)]
            client = Client(client_id=num, sparsity=0.2, tag=client_tag)
            clients.append(client)
        
        return clients
