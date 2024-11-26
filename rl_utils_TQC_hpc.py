import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import math

from rl_class_TQC_hpc import GlobalParams
from rl_opt_TQC_hpc import calculate_optimum
from rl_hyp_param_TQC_hpc import HypParams


def import_market_data(csvfile: str, type: str):  # , df: pd.DataFrame) # for future implementation
    """
        Import data of day-ahead prices for electricity
        :param csvfile: Time; <data>
        :param type: market data type
        :return arr: np.array with market data
    """

    file_path = os.path.dirname(__file__) + "/" + csvfile
    df = pd.read_csv(file_path, delimiter=";", decimal=".")
    df["Time"] = pd.to_datetime(df["Time"], format="%d-%m-%Y %H:%M")

    if type == "elec": # electricity price data
        arr = df["Day-Ahead-price [Euro/MWh]"].values.astype(float) / 10  # Convert Euro/MWh into ct/kWh
    elif type == "gas": # gas price data
        arr = df["THE_DA_Gas [Euro/MWh]"].values.astype(float) / 10   # Convert Euro/MWh into ct/kWh
    elif type == "eua": # EUA price data
        arr = df["EUA_CO2 [Euro/t]"].values.astype(float)
    else:
        assert False, "Market data type not specified appropriately [elec, gas, eua]!"

    return arr


def import_data(csvfile: str):
    """
        Import experimental methanation data for state changes
        :param csvfile - columns
        ["Time [s]", "T_cat [°C]", "n_h2 [mol/s]", "n_ch4 [mol/s]", "n_h2_res [mol/s]", "m_DE [kg/h]", "Pel [W]"]
        :return arr: np.array with operational data
    """
    # Import historic Data from csv file
    file_path = os.path.dirname(__file__) + "/" + csvfile
    df = pd.read_csv(file_path, delimiter=";", decimal=".")

    # Transform data to numpy array
    t = df["Time [s]"].values.astype(float)
    t_cat = df["T_cat [gradC]"].values.astype(float)
    n_h2 = df["n_h2 [mol/s]"].values.astype(float)
    n_ch4 = df["n_ch4 [mol/s]"].values.astype(float)
    n_h2_res = df["n_h2_res [mol/s]"].values.astype(float)
    m_h2o = df["m_DE [kg/h]"].values.astype(float)
    p_el = df["Pel [W]"].values.astype(float)
    arr = np.c_[t, t_cat, n_h2, n_ch4, n_h2_res, m_h2o, p_el]

    return arr


def load_data():
    """
    Loads historical market data and experimental data of methanation operation
    :return dict_price_data: dictionary with market data
            dict_op_data: dictionary with data of dynamic methanation operation
    """

    GLOBAL_PARAMS = GlobalParams()

    # Load historical market data for electricity, gas and EUA
    dict_price_data = {'el_price_train': import_market_data(GLOBAL_PARAMS.datafile_path_train_el, "elec"),
                       'el_price_cv': import_market_data(GLOBAL_PARAMS.datafile_path_cv_el, "elec"),
                       'el_price_test': import_market_data(GLOBAL_PARAMS.datafile_path_test_el, "elec"),
                       'gas_price_train': import_market_data(GLOBAL_PARAMS.datafile_path_train_gas, "gas"),
                       'gas_price_cv': import_market_data(GLOBAL_PARAMS.datafile_path_cv_gas, "gas"),
                       'gas_price_test': import_market_data(GLOBAL_PARAMS.datafile_path_test_gas, "gas"),
                       'eua_price_train': import_market_data(GLOBAL_PARAMS.datafile_path_train_eua, "eua"),
                       'eua_price_cv': import_market_data(GLOBAL_PARAMS.datafile_path_cv_eua, "eua"),
                       'eua_price_test': import_market_data(GLOBAL_PARAMS.datafile_path_test_eua, "eua")}

    # Load experimental methanation data for state changes
    dict_op_data = {'startup_cold': import_data(GLOBAL_PARAMS.datafile_path2),  # cold start
                    'startup_hot': import_data(GLOBAL_PARAMS.datafile_path3),  # hot start
                    'cooldown': import_data(GLOBAL_PARAMS.datafile_path4),  # cooldown
                    'standby_down': import_data(GLOBAL_PARAMS.datafile_path5),
                    # standby dataset for high temperatures to standby
                    'standby_up': import_data(GLOBAL_PARAMS.datafile_path6),
                    # standby dataset for low temperatures to standby
                    'op1_start_p': import_data(GLOBAL_PARAMS.datafile_path7),  # partial load - warming up
                    'op2_start_f': import_data(GLOBAL_PARAMS.datafile_path8),  # full load - warming up
                    'op3_p_f': import_data(GLOBAL_PARAMS.datafile_path9),  # Load change: Partial -> Full
                    'op4_p_f_p_5': import_data(GLOBAL_PARAMS.datafile_path10),
                    # Load change: Partial -> Full -> Partial (Return after 5 min)
                    'op5_p_f_p_10': import_data(GLOBAL_PARAMS.datafile_path11),
                    # Load change: Partial -> Full -> Partial (Return after 10 min)
                    'op6_p_f_p_15': import_data(GLOBAL_PARAMS.datafile_path12),
                    # Load change: Partial -> Full -> Partial (Return after 15 min)
                    'op7_p_f_p_22': import_data(GLOBAL_PARAMS.datafile_path13),
                    # Load change: Partial -> Full -> Partial (Return after 22 min)
                    'op8_f_p': import_data(GLOBAL_PARAMS.datafile_path14),  # Load change: Full -> Partial
                    'op9_f_p_f_5': import_data(GLOBAL_PARAMS.datafile_path15),
                    # Load change: Full -> Partial -> Full (Return after 5 min)
                    'op10_f_p_f_10': import_data(GLOBAL_PARAMS.datafile_path16),
                    # Load change: Full -> Partial -> Full (Return after 10 min)
                    'op11_f_p_f_15': import_data(GLOBAL_PARAMS.datafile_path17),
                    # Load change: Full -> Partial -> Full (Return after 15 min)
                    'op12_f_p_f_20': import_data(
                        GLOBAL_PARAMS.datafile_path18)}  # Load change: Full -> Partial -> Full (Return after 20 min)

    if GLOBAL_PARAMS.scenario == 2:  # fixed gas price market data
        dict_price_data['gas_price_train'] = np.ones(
            len(dict_price_data['gas_price_train'])) * GLOBAL_PARAMS.ch4_price_fix
        dict_price_data['gas_price_cv'] = np.ones(
            len(dict_price_data['gas_price_cv'])) * GLOBAL_PARAMS.ch4_price_fix
        dict_price_data['gas_price_test'] = np.ones(
            len(dict_price_data['gas_price_test'])) * GLOBAL_PARAMS.ch4_price_fix
    elif GLOBAL_PARAMS.scenario == 3:  # gas price and eua = 0
        dict_price_data['gas_price_train'] = np.zeros(len(dict_price_data['gas_price_train']))
        dict_price_data['gas_price_cv'] = np.zeros(len(dict_price_data['gas_price_cv']))
        dict_price_data['gas_price_test'] = np.zeros(len(dict_price_data['gas_price_test']))
        dict_price_data['eua_price_train'] = np.zeros(len(dict_price_data['eua_price_train']))
        dict_price_data['eua_price_cv'] = np.zeros(len(dict_price_data['eua_price_test']))
        dict_price_data['eua_price_test'] = np.zeros(len(dict_price_data['eua_price_test']))

    # For Reward level calculation -> Sets height of the reward penalty
    dict_price_data['el_price_reward_level'] = GLOBAL_PARAMS.r_0_values['el_price']
    dict_price_data['gas_price_reward_level'] = GLOBAL_PARAMS.r_0_values['gas_price']
    dict_price_data['eua_price_reward_level'] = GLOBAL_PARAMS.r_0_values['eua_price']

    return dict_price_data, dict_op_data


def preprocessing_rew(dict_price_data):
    """
    Data preprocessing including the computation of a potential reward, which signifies the maximum reward the
    Power-to-Gas plant can yield in either partial load [part_full_b... = 0] or full load [part_full_b... = 1]
    :param dict_price_data: dictionary with market data
    :return dict_pot_r_b: dictionary with potential reward [pot_rew...] and boolean reward identifier [part_full_b...]
    """

    GLOBAL_PARAMS = GlobalParams()

    # compute methanation operation data for theoretical optimum (ignoring dynamics)
    stats_dict_opt_train = calculate_optimum(dict_price_data['el_price_train'], dict_price_data['gas_price_train'],
                                             dict_price_data['eua_price_train'], "Train")
    stats_dict_opt_cv = calculate_optimum(dict_price_data['el_price_cv'], dict_price_data['gas_price_cv'],
                                            dict_price_data['eua_price_cv'], "CV")
    stats_dict_opt_test = calculate_optimum(dict_price_data['el_price_test'], dict_price_data['gas_price_test'],
                                            dict_price_data['eua_price_test'], "Test")
    stats_dict_opt_level = calculate_optimum(dict_price_data['el_price_reward_level'], dict_price_data['gas_price_reward_level'],
                                            dict_price_data['eua_price_reward_level'], "reward_Level")

    # Store data sets with future values of the potential reward on the two different load levels and
    # data sets of a boolean identifier of future values of the potential reward in a dictionary
    # if pot_rew_... <= 0:
    #   part_full_b_... = -1
    # else:
    #   if (pot_rew_... in full load) < (pot_rew... in partial load):
    #       part_full_b_... = 0
    #   else:
    #       part_full_b_... = 1
    dict_pot_r_b = {
        'pot_rew_train': stats_dict_opt_train['Meth_reward_stats'],
        'part_full_b_train': stats_dict_opt_train['partial_full_b'],
        'pot_rew_cv': stats_dict_opt_cv['Meth_reward_stats'],
        'part_full_b_cv': stats_dict_opt_cv['partial_full_b'],
        'pot_rew_test': stats_dict_opt_test['Meth_reward_stats'],
        'part_full_b_test': stats_dict_opt_test['partial_full_b'],
    }

    r_level = stats_dict_opt_level['Meth_reward_stats']
    # multiple_plots(stats_dict_opt_train, 3600, "Opt_Training_set_sen" + str(GLOBAL_PARAMS.scenario))
    # multiple_plots(stats_dict_opt_test, 3600, "Opt_Test_set_sen" + str(GLOBAL_PARAMS.scenario))

    return dict_pot_r_b, r_level


def preprocessing_array(dict_price_data, dict_pot_r_b):
    """
    Transforms dictionaries to np.arrays for computation purposes
    :param dict_price_data: market data
    :param dict_pot_r_b: potential reward and boolean reward identifier
    :return:    e_r_b_train/e_r_b_cv/e_r_b_test: (hourly values)
                    np.array which stores elec. price data, potential reward, and boolean identifier
                    Dimensions = [Type of data] x [No. of day-ahead values] x [historical values]
                        Type of data = [el_price, pot_rew, part_full_b]
                        No. of day-ahead values = GLOBAL_PARAMS.price_ahead
                        historical values = No. of values in the electricity price data set
                g_e_train/g_e_cv/g_e_test: (daily values)
                    np.array which stores gas and eua price data
                    Dimensions = [Type of data] x [No. of day-ahead values] x [historical values]
                        Type of data = [gas_price, pot_rew, part_full_b]
                        No. of day-ahead values = 2 (today and tomorrow)
                        historical values = No. of values in the gas/eua price data set
    """

    GLOBAL_PARAMS = GlobalParams()

    # Multi-Dimensional Array (3D) which stores day-ahead electricity price data as well as day-ahead potential reward
    # and boolean identifier for the entire training and test set
    # e.g. e_r_b_train[0, 5, 156] represents the future value of the electricity price [0,-,-] in 4 hours [-,5,-] at the
    # 156ths entry of the electricity price data set
    e_r_b_train = np.zeros((3, GLOBAL_PARAMS.price_ahead,
                            dict_price_data['el_price_train'].shape[0] - GLOBAL_PARAMS.price_ahead))
    e_r_b_cv = np.zeros((3, GLOBAL_PARAMS.price_ahead,
                           dict_price_data['el_price_cv'].shape[0] - GLOBAL_PARAMS.price_ahead))
    e_r_b_test = np.zeros((3, GLOBAL_PARAMS.price_ahead,
                           dict_price_data['el_price_test'].shape[0] - GLOBAL_PARAMS.price_ahead))

    for i in range(GLOBAL_PARAMS.price_ahead):
        e_r_b_train[0, i, :] = dict_price_data['el_price_train'][i:(-GLOBAL_PARAMS.price_ahead + i)]
        e_r_b_train[1, i, :] = dict_pot_r_b['pot_rew_train'][i:(-GLOBAL_PARAMS.price_ahead + i)]
        e_r_b_train[2, i, :] = dict_pot_r_b['part_full_b_train'][i:(-GLOBAL_PARAMS.price_ahead + i)]
        e_r_b_cv[0, i, :] = dict_price_data['el_price_cv'][i:(-GLOBAL_PARAMS.price_ahead + i)]
        e_r_b_cv[1, i, :] = dict_pot_r_b['pot_rew_cv'][i:(-GLOBAL_PARAMS.price_ahead + i)]
        e_r_b_cv[2, i, :] = dict_pot_r_b['part_full_b_cv'][i:(-GLOBAL_PARAMS.price_ahead + i)]
        e_r_b_test[0, i, :] = dict_price_data['el_price_test'][i:(-GLOBAL_PARAMS.price_ahead + i)]
        e_r_b_test[1, i, :] = dict_pot_r_b['pot_rew_test'][i:(-GLOBAL_PARAMS.price_ahead + i)]
        e_r_b_test[2, i, :] = dict_pot_r_b['part_full_b_test'][i:(-GLOBAL_PARAMS.price_ahead + i)]

    # Multi-Dimensional Array (3D) which stores day-ahead gas and eua price data for the entire training and test set
    g_e_train = np.zeros((2, 2, dict_price_data['gas_price_train'].shape[0] - 1))
    g_e_cv = np.zeros((2, 2, dict_price_data['gas_price_cv'].shape[0] - 1))
    g_e_test = np.zeros((2, 2, dict_price_data['gas_price_test'].shape[0] - 1))

    g_e_train[0, 0, :] = dict_price_data['gas_price_train'][:-1]
    g_e_train[1, 0, :] = dict_price_data['eua_price_train'][:-1]
    g_e_cv[0, 0, :] = dict_price_data['gas_price_cv'][:-1]
    g_e_cv[1, 0, :] = dict_price_data['eua_price_cv'][:-1]
    g_e_test[0, 0, :] = dict_price_data['gas_price_test'][:-1]
    g_e_test[1, 0, :] = dict_price_data['eua_price_test'][:-1]
    g_e_train[0, 1, :] = dict_price_data['gas_price_train'][1:]
    g_e_train[1, 1, :] = dict_price_data['eua_price_train'][1:]
    g_e_cv[0, 1, :] = dict_price_data['gas_price_cv'][1:]
    g_e_cv[1, 1, :] = dict_price_data['eua_price_cv'][1:]
    g_e_test[0, 1, :] = dict_price_data['gas_price_test'][1:]
    g_e_test[1, 1, :] = dict_price_data['eua_price_test'][1:]

    return e_r_b_train, e_r_b_cv, e_r_b_test, g_e_train, g_e_cv, g_e_test


def define_episodes(dict_price_data, seed_train):
    """
    Defines specifications for training and evaluation episodes
    :param dict_price_data: dictionary with market data
    :param seed_train: random seed for training
    :return eps_sim_steps_train: Number of steps in the training set per episode
            eps_sim_steps_test: Number of steps in the test set per episode
            eps_ind: contains indexes of the training subsets
            total_n_steps: total number of steps for all workers together
            n_eps_loops: No. of training subsets times No. of loops
    """

    GLOBAL_PARAMS = GlobalParams()
    HYPER_PARAMS = HypParams()

    print("Define episodes and step size limits...")
    # No. of days in the total training set
    train_len_d = GLOBAL_PARAMS.train_len_d
    # No. of days in the test set ("-1" excludes the day-ahead overhead)
    cv_len_d = len(dict_price_data['gas_price_cv']) - 1
    test_len_d = len(dict_price_data['gas_price_test']) - 1

    # Split up the entire training set into several smaller subsets which represents an own episodes
    if train_len_d % HYPER_PARAMS.eps_len_d == 0:
        n_eps = int(train_len_d / HYPER_PARAMS.eps_len_d)  # number of training subsets
    else:
        assert False, "No. of days in the training set (train_len_d) should be a factor of the episode length (eps_len_d)!"

    eps_len = 24 * 3600 * HYPER_PARAMS.eps_len_d  # episode length in seconds

    # Number of steps in train and test set per episode
    eps_sim_steps_train = int(eps_len / HYPER_PARAMS.sim_step)
    eps_sim_steps_cv = int(24 * 3600 * cv_len_d / HYPER_PARAMS.sim_step)
    eps_sim_steps_test = int(24 * 3600 * test_len_d / HYPER_PARAMS.sim_step)

    # Define total number of steps for all workers together
    num_loops = GLOBAL_PARAMS.num_loops  # number of loops over the total training set
    overhead = 2000  # small overhead for training
    total_n_steps = int(math.ceil(eps_sim_steps_train * n_eps * num_loops) + overhead)
    print("--- Total number of training steps =", total_n_steps)
    print("--- Training steps per episode =", eps_sim_steps_train)
    print("--- Steps in the evaluation set =", eps_sim_steps_test)

    eps_ind = rand_eps_ind(train_len_d, HYPER_PARAMS.eps_len_d, n_eps, num_loops, seed_train)

    n_eps_loops = n_eps * num_loops

    return eps_sim_steps_train, eps_sim_steps_cv, eps_sim_steps_test, eps_ind, total_n_steps, n_eps_loops


def rand_eps_ind(train_len_d: int, eps_len_d: int, n_eps: int, num_loops: int, seed: int):
    """
    The agent can either use the total training set in one episode (train_len_d == eps_len_d) or
    divide the total training set into smaller subsets (train_len_d_i > eps_len_d). In the latter case, the
    subsets where selected randomly
    :param train_len_d: Total number of days in the training set
    :param eps_len_d: Number of days in one training episode
    :param n_eps: Number of training subsets
    :param num_loops: Number of loops over the total training set
    :param seed: random seed of the trainings set
    :return: eps_ind: contains indexes of the training subsets
    """

    np.random.seed(seed)

    overhead_factor = 3     # to account for randomn selection of ep_index of the different processes in multiprocessing

    if train_len_d == eps_len_d:
        eps_ind = np.zeros(int(n_eps*num_loops*overhead_factor))
    elif train_len_d > eps_len_d:
        # random selection with sampling with replacement
        num_ep = np.linspace(start=0, stop=n_eps-1, num=n_eps)
        random_ep = np.zeros((num_loops*overhead_factor, n_eps))
        for i in range(num_loops*overhead_factor):
            random_ep[i, :] = num_ep
            np.random.shuffle(random_ep[i, :])
        eps_ind = random_ep.reshape(int(n_eps*num_loops*overhead_factor)).astype(int)
    else:
        assert False, "train_len_d >= eps_len_d!"

    return eps_ind


def dict_env_kwargs(eps_ind, e_r_b, g_e, dict_op_data, eps_sim_steps, n_eps, r_level, type="train"):
    """
    Returns global model parameters and hyper parameters applied in the PtG environment as a dictionary
    :param eps_ind: contains indexes of the training subsets
    :param e_r_b: np.array which stores elec. price data, potential reward, and boolean identifier
    :param g_e: np.array which stores gas and eua price data
    :param dict_op_data: dictionary with potential reward [pot_rew...] and boolean reward identifier [part_full_b...]
    :param eps_sim_steps: training/test episode length
    :param n_eps: No. of training subsets
    :param type: specifies either the training set "train" or the cv/ test set "cv_test"
    :return: env_kwargs: dictionary with global parameters and hyperparameters
    """

    GLOBAL_PARAMS = GlobalParams()
    HYPER_PARAMS = HypParams()

    env_kwargs = {}

    env_kwargs["ptg_state_space['standby']"] = GLOBAL_PARAMS.ptg_state_space['standby']
    env_kwargs["ptg_state_space['cooldown']"] = GLOBAL_PARAMS.ptg_state_space['cooldown']
    env_kwargs["ptg_state_space['startup']"] = GLOBAL_PARAMS.ptg_state_space['startup']
    env_kwargs["ptg_state_space['partial_load']"] = GLOBAL_PARAMS.ptg_state_space['partial_load']
    env_kwargs["ptg_state_space['full_load']"] = GLOBAL_PARAMS.ptg_state_space['full_load']

    env_kwargs["noise"] = GLOBAL_PARAMS.noise
    env_kwargs["parallel"] = GLOBAL_PARAMS.parallel
    env_kwargs["eps_ind"] = eps_ind                     # differ in train and test set
    env_kwargs["eps_len_d"] = HYPER_PARAMS.eps_len_d
    env_kwargs["sim_step"] = HYPER_PARAMS.sim_step
    env_kwargs["time_step_op"] = GLOBAL_PARAMS.time_step_op
    env_kwargs["price_ahead"] = GLOBAL_PARAMS.price_ahead
    env_kwargs["n_eps_loops"] = n_eps

    env_kwargs["e_r_b"] = e_r_b                         # differ in train and test set
    env_kwargs["g_e"] = g_e                             # differ in train and test set

    env_kwargs["dict_op_data['startup_cold']"] = dict_op_data['startup_cold']
    env_kwargs["dict_op_data['startup_hot']"] = dict_op_data['startup_hot']
    env_kwargs["dict_op_data['cooldown']"] = dict_op_data['cooldown']
    env_kwargs["dict_op_data['standby_down']"] = dict_op_data['standby_down']
    env_kwargs["dict_op_data['standby_up']"] = dict_op_data['standby_up']
    env_kwargs["dict_op_data['op1_start_p']"] = dict_op_data['op1_start_p']
    env_kwargs["dict_op_data['op2_start_f']"] = dict_op_data['op2_start_f']
    env_kwargs["dict_op_data['op3_p_f']"] = dict_op_data['op3_p_f']
    env_kwargs["dict_op_data['op4_p_f_p_5']"] = dict_op_data['op4_p_f_p_5']
    env_kwargs["dict_op_data['op5_p_f_p_10']"] = dict_op_data['op5_p_f_p_10']
    env_kwargs["dict_op_data['op6_p_f_p_15']"] = dict_op_data['op6_p_f_p_15']
    env_kwargs["dict_op_data['op7_p_f_p_22']"] = dict_op_data['op7_p_f_p_22']
    env_kwargs["dict_op_data['op8_f_p']"] = dict_op_data['op8_f_p']
    env_kwargs["dict_op_data['op9_f_p_f_5']"] = dict_op_data['op9_f_p_f_5']
    env_kwargs["dict_op_data['op10_f_p_f_10']"] = dict_op_data['op10_f_p_f_10']
    env_kwargs["dict_op_data['op11_f_p_f_15']"] = dict_op_data['op11_f_p_f_15']
    env_kwargs["dict_op_data['op12_f_p_f_20']"] = dict_op_data['op12_f_p_f_20']

    env_kwargs["scenario"] = GLOBAL_PARAMS.scenario

    env_kwargs["convert_mol_to_Nm3"] = GLOBAL_PARAMS.convert_mol_to_Nm3
    env_kwargs["H_u_CH4"] = GLOBAL_PARAMS.H_u_CH4
    env_kwargs["H_u_H2"] = GLOBAL_PARAMS.H_u_H2
    env_kwargs["dt_water"] = GLOBAL_PARAMS.dt_water
    env_kwargs["cp_water"] = GLOBAL_PARAMS.cp_water
    env_kwargs["rho_water"] = GLOBAL_PARAMS.rho_water
    env_kwargs["Molar_mass_CO2"] = GLOBAL_PARAMS.Molar_mass_CO2
    env_kwargs["Molar_mass_H2O"] = GLOBAL_PARAMS.Molar_mass_H2O
    env_kwargs["h_H2O_evap"] = GLOBAL_PARAMS.h_H2O_evap
    env_kwargs["eeg_el_price"] = GLOBAL_PARAMS.eeg_el_price
    env_kwargs["heat_price"] = GLOBAL_PARAMS.heat_price
    env_kwargs["o2_price"] = GLOBAL_PARAMS.o2_price
    env_kwargs["water_price"] = GLOBAL_PARAMS.water_price
    env_kwargs["min_load_electrolyzer"] = GLOBAL_PARAMS.min_load_electrolyzer
    env_kwargs["max_h2_volumeflow"] = GLOBAL_PARAMS.max_h2_volumeflow
    env_kwargs["eta_BHKW"] = GLOBAL_PARAMS.eta_BHKW

    env_kwargs["t_cat_standby"] = GLOBAL_PARAMS.t_cat_standby
    env_kwargs["t_cat_startup_cold"] = GLOBAL_PARAMS.t_cat_startup_cold
    env_kwargs["t_cat_startup_hot"] = GLOBAL_PARAMS.t_cat_startup_hot
    env_kwargs["time1_start_p_f"] = GLOBAL_PARAMS.time1_start_p_f
    env_kwargs["time2_start_f_p"] = GLOBAL_PARAMS.time2_start_f_p
    env_kwargs["time_p_f"] = GLOBAL_PARAMS.time_p_f
    env_kwargs["time_f_p"] = GLOBAL_PARAMS.time_f_p
    env_kwargs["time1_p_f_p"] = GLOBAL_PARAMS.time1_p_f_p
    env_kwargs["time2_p_f_p"] = GLOBAL_PARAMS.time2_p_f_p
    env_kwargs["time23_p_f_p"] = GLOBAL_PARAMS.time23_p_f_p
    env_kwargs["time3_p_f_p"] = GLOBAL_PARAMS.time3_p_f_p
    env_kwargs["time34_p_f_p"] = GLOBAL_PARAMS.time34_p_f_p
    env_kwargs["time4_p_f_p"] = GLOBAL_PARAMS.time4_p_f_p
    env_kwargs["time45_p_f_p"] = GLOBAL_PARAMS.time45_p_f_p
    env_kwargs["time5_p_f_p"] = GLOBAL_PARAMS.time5_p_f_p
    env_kwargs["time1_f_p_f"] = GLOBAL_PARAMS.time1_f_p_f
    env_kwargs["time2_f_p_f"] = GLOBAL_PARAMS.time2_f_p_f
    env_kwargs["time23_f_p_f"] = GLOBAL_PARAMS.time23_f_p_f
    env_kwargs["time3_f_p_f"] = GLOBAL_PARAMS.time3_f_p_f
    env_kwargs["time34_f_p_f"] = GLOBAL_PARAMS.time34_f_p_f
    env_kwargs["time4_f_p_f"] = GLOBAL_PARAMS.time4_f_p_f
    env_kwargs["time45_f_p_f"] = GLOBAL_PARAMS.time45_f_p_f
    env_kwargs["time5_f_p_f"] = GLOBAL_PARAMS.time5_f_p_f
    env_kwargs["i_fully_developed"] = GLOBAL_PARAMS.i_fully_developed
    env_kwargs["j_fully_developed"] = GLOBAL_PARAMS.j_fully_developed

    env_kwargs["t_cat_startup_cold"] = GLOBAL_PARAMS.t_cat_startup_cold
    env_kwargs["t_cat_startup_hot"] = GLOBAL_PARAMS.t_cat_startup_hot

    env_kwargs["rew_l_b"] = np.min(e_r_b[1, 0, :])
    env_kwargs["rew_u_b"] = np.max(e_r_b[1, 0, :])
    env_kwargs["T_l_b"] = GLOBAL_PARAMS.T_l_b
    env_kwargs["T_u_b"] = GLOBAL_PARAMS.T_u_b
    env_kwargs["h2_l_b"] = GLOBAL_PARAMS.h2_l_b
    env_kwargs["h2_u_b"] = GLOBAL_PARAMS.h2_u_b
    env_kwargs["ch4_l_b"] = GLOBAL_PARAMS.ch4_l_b
    env_kwargs["ch4_u_b"] = GLOBAL_PARAMS.ch4_u_b
    env_kwargs["h2_res_l_b"] = GLOBAL_PARAMS.h2_res_l_b
    env_kwargs["h2_res_u_b"] = GLOBAL_PARAMS.h2_res_u_b
    env_kwargs["h2o_l_b"] = GLOBAL_PARAMS.h2o_l_b
    env_kwargs["h2o_u_b"] = GLOBAL_PARAMS.h2o_u_b
    env_kwargs["heat_l_b"] = GLOBAL_PARAMS.heat_l_b
    env_kwargs["heat_u_b"] = GLOBAL_PARAMS.heat_u_b

    env_kwargs["eps_sim_steps"] = eps_sim_steps         # differ in train and test set

    if type == "train":
        env_kwargs["state_change_penalty"] = HYPER_PARAMS.state_change_penalty
    elif type == "cv_test":
        env_kwargs["state_change_penalty"] = 0.0        # no state change penalty during validation

    env_kwargs["reward_level"] = r_level
    env_kwargs["action_type"] = GLOBAL_PARAMS.action_type

    return env_kwargs


def multiple_plots(stats_dict: dict, time_step_size: int, plot_name: str):
    """
    Creates a plot with multiple subplots of the time series and the methanation operation according to the agent
    :param stats_dict: dictionary with the prediction results
    :param time_step_size: time step size in the simulation
    :param plot_name: plot title
    :return
    """

    part_full_b = stats_dict['partial_full_b']
    time_sim = stats_dict['steps_stats'] * time_step_size * 1 / 3600 / 24  # in days
    time_sim_profit =  np.linspace(0, int(len(part_full_b))-1, num=int(len(part_full_b))) / 24
    profit_series = part_full_b * 50

    fig, axs = plt.subplots(8, 1, figsize=(10, 6), sharex=True, sharey=False)
    axs[0].plot(time_sim, stats_dict['el_price_stats'], label='el_price')
    axs[0].plot(time_sim, stats_dict['gas_price_stats'], 'g', label='gas_price')
    axs[0].plot(time_sim_profit, profit_series, 'r', label='profitable')
    # axs[0].set_ylim([0, 5.5])
    axs[0].set_ylabel('ct/kWh')
    axs[0].legend(loc="upper right", fontsize='x-small')
    axs0_1 = axs[0].twinx()
    axs0_1.plot(time_sim, stats_dict['eua_price_stats'], 'k', label='eua_price')
    axs0_1.set_ylabel('eua_price [€/t]')
    axs0_1.legend(loc="lower right", fontsize='x-small')
    axs[1].plot(time_sim, stats_dict['Meth_State_stats'], 'b', label='state')
    axs[1].plot(time_sim, stats_dict['Meth_Action_stats'], 'g', label='action')
    axs[1].plot(time_sim, stats_dict['Meth_Hot_Cold_stats'], 'k', label='hot/cold-status')
    # axs[1].set_ylim([0, 12])
    axs[1].set_ylabel('status')
    axs[1].legend(loc="upper right", fontsize='x-small')
    axs[2].plot(time_sim, stats_dict['Meth_T_cat_stats'], 'k', label='T_Cat')
    # axs[2].plot(time_sim, stats_dict['Meth_T_cat_stats'], 'k', marker='o', markersize=2)
    # axs[2].set_ylim([0, 600])
    axs[2].set_ylabel('°C')
    axs[2].legend(loc="upper right", fontsize='x-small')
    axs[3].plot(time_sim, stats_dict['Meth_H2_flow_stats'], 'b', label='H2')
    axs[3].plot(time_sim, stats_dict['Meth_CH4_flow_stats'], 'g', label='CH4')
    # axs[3].set_ylim([0, 0.025])
    axs[3].set_ylabel('mol/s')
    axs[3].legend(loc="upper right", fontsize='x-small')
    axs[4].plot(time_sim, stats_dict['Meth_H2O_flow_stats'], label='H2O')
    # axs[4].set_ylim([0, 0.72])
    axs[4].set_ylabel('kg/h')
    axs[4].legend(loc="upper right", fontsize='x-small')
    axs[5].plot(time_sim, stats_dict['Meth_el_heating_stats'], label='P_el_heat')
    # axs[5].set_ylim([-10, 2000])
    axs[5].set_ylabel('W')
    axs[5].legend(loc="upper right", fontsize='x-small')
    axs[6].plot(time_sim, stats_dict['Meth_ch4_revenues_stats'], 'g', label='CH4')
    axs[6].plot(time_sim, stats_dict['Meth_steam_revenues_stats'], 'b', label='H2O')
    axs[6].plot(time_sim, stats_dict['Meth_o2_revenues_stats'], 'lightgray', label='O2')
    axs[6].plot(time_sim, stats_dict['Meth_elec_costs_heating_stats'], 'k', label='P_el_heating')
    axs[6].plot(time_sim, stats_dict['Meth_elec_costs_electrolyzer_stats'], 'r', label='P_el_lyzer')
    axs[6].set_ylabel('ct/h')
    axs[6].legend(loc="upper right", fontsize='x-small')
    axs[7].plot(time_sim, stats_dict['Meth_reward_stats'], 'g', label='Reward')
    axs[7].set_ylabel('r [ct/h]')
    axs[7].set_xlabel('Time [d]')
    # axs[7].set_xlim([-10, 10000])
    axs[7].legend(loc="upper right", fontsize='x-small')
    axs7_1 = axs[7].twinx()
    axs7_1.plot(time_sim, stats_dict['Meth_cum_reward_stats'], 'k', label='Cumulative Reward')
    axs7_1.set_ylabel('cum_r [ct]')
    axs7_1.legend(loc="lower right", fontsize='x-small')

    fig.suptitle(" Alg:" + plot_name + "\n Rew:" + str(np.round(stats_dict['Meth_cum_reward_stats'][-1], 0)))
    plt.savefig('plots/' + plot_name + '_plot.png')
    print("Reward =", stats_dict['Meth_cum_reward_stats'][-1])

    # plt.show()
    plt.close()


def multiple_plots_4(stats_dict: dict, time_step_size: int, plot_name: str, potential_profit: np.array, part_full: np.array):
    """
    Creates a plot with multiple subplots of the time series and the methanation operation according to the agent
    :param stats_dict: dictionary with the prediction results
    :param time_step_size: time step size in the simulation
    :param plot_name: plot title
    :param profit: denotes periods with potential profit (No Profit = 0, Profit = 10)
    :return:
    """

    time_sim = stats_dict['steps_stats'] * time_step_size * 1 / 3600 / 24  # in days
    time_sim_profit =  np.linspace(0, int(len(profit))-1, num=int(len(profit))) / 24
    profit_series = 1+ profit * 4

    P_PtG = stats_dict['Meth_H2_flow_stats'] / 0.0198

    plt.rcParams.update({'font.size': 16})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    pot_rew = potential_profit * time_step_size / 3600

    rew_zero = np.zeros((len(time_sim),))
    part_full_adapted = np.zeros((len(time_sim),))
    for i in range(len(time_sim)):
        if part_full[i] == -1:
            part_full_adapted[i] = 2
        elif part_full[i] == 0:
            part_full_adapted[i] = 4
        else:
            part_full_adapted[i] = 5

    fig, axs = plt.subplots(1, 1, figsize=(14, 6), sharex=True, sharey=False)
    axs.plot(time_sim, stats_dict['Meth_State_stats'], 'b', label='state')
    axs.plot(time_sim, part_full_adapted, 'k', linestyle="dotted", label='state')
    axs.set_yticks([1,2,3,4,5])
    axs.set_yticklabels(['Standby', 'Cooldown/Off', 'Startup', 'Partial Load', 'Full Load'])
    # axs[1].set_ylim([0, 12])
    axs.set_ylabel(' ')
    axs.legend(loc="upper left", fontsize='small') #, bbox_to_anchor = (0.0, 0.0), ncol = 1, fancybox = True, shadow = True)
    axs.grid(axis='y', linestyle='dashed')
    axs0_1 = axs.twinx()
    axs0_1.plot(time_sim, stats_dict['Meth_reward_stats'], color='g', label='Reward')
    axs0_1.plot(time_sim, pot_rew, color='lawngreen', linestyle='dotted', label='Potential Reward')
    axs0_1.plot(time_sim, rew_zero, color='grey', linestyle='dashed')
    axs0_1.set_ylabel('reward')
    axs0_1.set_xlabel('Time [d]')
    axs0_1.set_yticks([0, 10, 20])
    # axs3_1 = axs[3].twinx()
    # axs3_1.plot(time_sim, stats_dict['Meth_cum_reward_stats'], 'k', label='Cumulative Reward')
    # axs3_1.set_ylabel('cum. reward')
    # # axs3_1.set_yticks([0, 1000, 2000])
    # axs[3].legend(loc="upper left", fontsize='small') #, bbox_to_anchor=(0.0, 0.0), ncol=1, fancybox=True, shadow=True)
    # axs3_1.legend(loc="upper right", fontsize='small') #, bbox_to_anchor = (0.0, 0.0), ncol = 1, fancybox = True, shadow = True)

    # box = axs0_1.get_position()
    # axs0_1.set_position([box.x0 * 1.1, box.y0 * 1.05, box.width, box.height])
    # box = axs[1].get_position()
    # axs[1].set_position([box.x0 * 1.1, box.y0 * 1.05, box.width, box.height])
    # box = axs2_1.get_position()
    # axs2_1.set_position([box.x0 * 1.1, box.y0 * 1.05, box.width, box.height])
    # box = axs3_1.get_position()
    # axs3_1.set_position([box.x0 * 1.1, box.y0, box.width, box.height])

    fig.suptitle(" Alg:" + plot_name + "\n Rew:" + str(np.round(stats_dict['Meth_cum_reward_stats'][-1], 0)))
    plt.savefig('plots/' + plot_name + '_plot.png')
    # print("Reward =", stats_dict['Meth_cum_reward_stats'][-1])

    plt.close()
    # plt.show()

