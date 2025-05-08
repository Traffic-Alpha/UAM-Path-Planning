import gymnasium as gym
from env_utils.aircraft_snir_env import ACSNIREnvironment
from env_utils.ac_wrapper import ACEnvWrapper
from stable_baselines3.common.monitor import Monitor

def make_env(
        num_seconds:int,sumo_cfg:str,use_gui:bool,
        net_file:str, snir_files:dict,
        log_file:str, aircraft_inits:dict,env_index:int,
        passenger_seq:list,snir_min:int,
        ):
    def _init() -> gym.Env:
        ac_env = ACSNIREnvironment(
            sumo_cfg=sumo_cfg,
            num_seconds=num_seconds,
            net_file=net_file,
            radio_map_files=snir_files,
            aircraft_inits=aircraft_inits,
            use_gui=use_gui
        )
        ac_wrapper = ACEnvWrapper(env=ac_env, aircraft_inits=aircraft_inits, passenger_info=passenger_seq, snir_min=snir_min)
        ac_env = Monitor(ac_wrapper, filename=f'{log_file}/{env_index}')
        return ac_env

    return _init