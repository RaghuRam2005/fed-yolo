#server.py
import random
from typing import List

from ultralytics.engine.model import Model

from .strategy import Strategy
from .client import Client

class Server:
    def __init__(self, communication_rounds:int, model:Model, strategy:Strategy, num_nodes:int):
        self.communication_rounds = communication_rounds
        self.global_model = model
        self.global_state = self.global_model.model.model.state_dict()
        self.strategy = strategy
        self.num_nodes = num_nodes
    
    def create_clients(self, tag_list:List) -> List[Client]:
        clients = []
        for num in range(self.num_nodes):
            if tag_list:
                tag = tag_list[random.randint(0, len(tag_list)-1)]
                client = Client(client_id=num, sparsity=0.2, tag=tag)
            else:
                client = Client(client_id=num, sparsity=0.2)
            clients.append(client)
        
        return clients