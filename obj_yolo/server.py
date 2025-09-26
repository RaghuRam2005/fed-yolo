#server.py
from typing import List

from ultralytics.engine.model import Model

from .strategy import Strategy
from .client import Client

class Server:
    def __init__(self, communication_rounds:int, model:Model, strategy:Strategy, num_nodes:int):
        self.communication_rounds = communication_rounds
        self.global_model = model
        self.global_state = self.global_model.state_dict()
        self.strategy = strategy
        self.num_nodes = num_nodes
    
    def create_clients(self) -> List[Client]:
        clients = []
        for num in range(self.num_nodes):
            client = Client(client_id=num, sparsity=0.2)
            clients.append(client)
        
        return clients