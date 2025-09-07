import logging
from dataclasses import dataclass

from torch import nn

from server import Server
from client import Client
from dataset import load_data
from strategy import Strategy

@dataclass
class Simulation:
    """
    configuration data class for simulation
    """
    server:Server
    num_supernodes:int = 2
    epochs:int = None
    lam:float = None
    yolo_config:str = None
    base_path:str
    client_base_path:str
    client_data_count:int

def run_simulation(simulation:Simulation):
    """
    A simple method to run simulation of federated client instead of starting a server

    Args:
        simulation (Simulation): instance of simulation data class
    """
    strategy = simulation.server.strategy
    server = simulation.server
    logging.info("parameter info at global initial state:")
    backbone = server.global_model.model[:11]
    total_params = sum(p.numel() for p in backbone.parameters())
    bn_params = sum(m.weight.numel() for m in backbone.modules()
                   if isinstance(m, nn.BatchNorm2d) and m.weight is not None)
    logging.info(f"Total parameters: {total_params}")
    logging.info(f"BN parameters   : {bn_params}")
    global_state = server.global_state
    clients = strategy.configure_fit(simulation=simulation, global_state=global_state)
    for _ in range(server.communication_rounds):
        data_paths = []
        results = []
        for client in clients:
            client_fn = Client(client_manager=client)
            logging.info(f"client training started for {client.client_id}")
            data_path = load_data(
                base_path=simulation.base_path,
                client_base_path=simulation.client_base_path,
                client_id=client.client_id,
                client_data_count=simulation.client_data_count,
            )
            data_paths.append(data_path)
            logging.info(f"prepared data for {client.client_id}")
            logging.info(f"data path: {data_path}")
            res = client_fn.fit(data_path=data_path)
            logging.info(f"client {client.client_id} training is complete")
            logging.info(f"Training results for {client.client_id}: ")
            logging.info(f"mAP50-95: {res.results["mAP50-95"]}")
            logging.info(f"map50   : {res.results["map50"]}")
            logging.info(f"map75   : {res.results["map75"]}")
            backbone = res.delta["backbone_delta"]
            total_params = sum(p.numel() for p in backbone.parameters() if p != 0)
            bn_params = sum(m.weight.numel() for m in backbone.modules()
                        if isinstance(m, nn.BatchNorm2d) and m.weight is not None)
            logging.info(f"Total parameters: {total_params}")
            logging.info(f"BN parameters   : {bn_params}")
            results.append(res)
        logging.info("Aggregation started")
        agg_state = strategy.aggregate_fit(results=results, global_state=global_state)
        server.update_global_state(new_state=agg_state)
        logging.info("Aggregation Completed and updated global model")
        for client, data_path in zip(clients, data_paths):
            client_fn = Client(client_manager=client)
            logging.info(f"Validation started for {client.client_id}")
            agg_params_list = [v.cpu().numpy() for v in agg_state.values()]
            val_res = client_fn.evaluate(
                paramaters=agg_params_list, data_path=data_path
            )
            logging.info(f"Validation Complete for {client.client_id}")
            logging.info(f"validation results for {client.client_id}: ")
            logging.info(f"mAP50-95: {val_res["mAP50-95"]}")
            logging.info(f"map50   : {val_res["map50"]}")
            logging.info(f"map75   : {val_res["map75"]}")
            client_fn.update_client_model(agg_params_list)
            strategy.update_sparsity(manager=client)
        logging.info("Completed a communication round")
        logging.info("starting another round")
        
    logging.info("simulation is complete")

if __name__ == "__main__":
    yolo_config = "C:\\Users\\lingu\\study\\projects\\obj_yolo\\yolo_config\\yolo11.yaml"
    base_path = "C:\\Users\\lingu\\study\\projects\\obj_yolo\\base_data\\training"
    client_base_path = "C:\\Users\\lingu\\study\\projects\\obj_yolo\\prepared_data\\clients"

    num_supernodes = 3
    communication_rounds = 2
    lam = 1e-5
    epochs = 10

    strategy = Strategy(
        min_clients_aggregation=2
    )

    server = Server(
        strategy=strategy,
        yolo_config=yolo_config,
        communication_rounds=2
    )

    logging.info("Starting the simulation, with following configuration:")
    logging.info(f"communication rounds: {communication_rounds}")
    logging.info(f"l1 lambda: {lam}")
    logging.info(f"epochs: {epochs}")

    simulation = Simulation(
        server=server,
        num_supernodes=num_supernodes,
        epochs=epochs,
        lam=lam,
        yolo_config=yolo_config,
        base_path=base_path,
        client_base_path=client_base_path,
        client_data_count=1000
    )

    run_simulation(
        simulation=simulation
    )
