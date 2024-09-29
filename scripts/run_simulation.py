import time
import numpy as np
import numba as nb
import random
from pathlib import Path
import os
import yaml
from pathlib import Path

import june
from june.hdf5_savers import generate_world_from_hdf5
from june.interaction import Interaction
from june.infection import InfectionSelector
from argparse import ArgumentParser

from june.groups.travel import Travel
from june.groups.leisure import generate_leisure_for_config
from june.simulator import Simulator
from june.infection_seed import InfectionSeed
from june.policy import Policies

from june import paths
from june.records import Record
import datetime

def set_random_seed(seed=999):
    """
    Sets global seeds for testing in numpy, random, and numbaized numpy.
    """
    @nb.njit(cache=True)
    def set_seed_numba(seed):
        random.seed(seed)
        return np.random.seed(seed)
    print(seed)
    np.random.seed(seed)
    set_seed_numba(seed)
    random.seed(seed)
    return


def generate_simulator(opts):

    output_folder = Path(opts.output_folder)
    config_path = paths.configs_path / "config_example_complete_second_wave.yaml"

    # find thw folder "worlds/" that contains our presaved world as an hdf5 file
    world_folder = paths.find_default("worlds")
    with open(paths.configs_path / "world_yaml.yaml") as f:
        world_config = yaml.load(f, Loader=yaml.FullLoader)

    world = generate_world_from_hdf5(world_folder / world_config['world_file'])
    leisure = generate_leisure_for_config(world, config_path)

    record = Record(record_path=output_folder, record_static_data=True)
    record.static_data(world=world)

    # health index and infection selecctor
    # The InfectionSeed class has no from_file() method, thats why we need to read the config 
    # separately and build an instance with init()
    with open(paths.configs_path / opts.path_to_config / "infection/seed_yaml.yaml") as f:
        seed_config = yaml.load(f, Loader=yaml.FullLoader)
	
    seed_strength = float(seed_config["seed_strength"])

    infection_selector = InfectionSelector.from_file()
    infection_seed = InfectionSeed(
        world=world,
        infection_selector=infection_selector,
        seed_strength=seed_strength,
        path_to_csv = paths.data_path / seed_config['path_to_csv']
    )
    #infection_seed.unleash_virus(population=world.people, n_cases=seed_config['n_cases'])
    infection_seed.min_date=datetime.datetime.strptime(seed_config['min_date'],"%Y-%m-%d")
    infection_seed.max_date=datetime.datetime.strptime(seed_config['max_date'],"%Y-%m-%d")

    # interaction
    interaction = Interaction.from_file(
        population=world.people,
        config_filename=paths.configs_path / opts.path_to_config / "interaction/interaction.yaml"
        )  
    # policies
    policies = Policies.from_file(
        config_file=paths.configs_path / opts.path_to_config / "policy/policy.yaml"
        )


    # create simulator
    travel = Travel(
        city_super_areas_filename=paths.data_path
        / world_config['city_super_areas_filename'],
        city_stations_filename=paths.configs_path
        / world_config['city_stations_filename'], 
        commute_config_filename=paths.configs_path
        / world_config['commute_config_filename'],
    )

    checkpoint_output_folder = Path(output_folder) / "checkpoints"
    checkpoint_output_folder.mkdir(parents=True, exist_ok=True)
    
    simulator = Simulator.from_file(
        world=world,
        policies=policies,
        interaction=interaction,
        leisure=leisure,
        travel=travel,
        infection_selector=infection_selector,
        infection_seed=infection_seed,
        config_filename=config_path,
        record=record,
        checkpoint_save_path=checkpoint_output_folder
    )
    print("simulator ready to go")
    return simulator


def run_simulator(simulator):
    t1 = time.time()
    simulator.run()
    t2 = time.time()
    print(f"Simulation took {(t2-t1)/60:.1f} minutes")


if __name__ == "__main__":

    print(paths.configs_path)
    argparser = ArgumentParser()
    argparser.add_argument("--seed", type=int, default=999)
    argparser.add_argument("--path-to-config", type=str)
    argparser.add_argument("--output-folder", default="results/")
    opts = argparser.parse_args()

        
    set_random_seed(opts.seed)
    simulator = generate_simulator(opts)
    run_simulator(simulator)

    
    
        






