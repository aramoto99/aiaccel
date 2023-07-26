# import copy
# from unittest.mock import patch

# import numpy as np
# import pytest

# from aiaccel.optimizer import NelderMead
# from aiaccel.parameter import HyperParameterConfiguration
# from aiaccel.storage import Storage
# from aiaccel.workspace import Workspace
# from tests.base_test import BaseTest



# def test_nelder_mead_parameters(load_test_config):
#     debug = False
#     config = load_test_config()
#     params = HyperParameterConfiguration(
#         # config.get('optimize', 'parameters')
#         config.optimize.parameters
#     )
#     initial_parameters = None
#     rng = np.random.RandomState(0)
#     nelder_mead = NelderMead(
#         params.get_parameter_list(), initial_parameters=initial_parameters,
#         iteration=100,
#         maximize=(config.optimize.goal[0].value.lower() == 'maximize'),
#         rng=rng
#     )

#     c_max = 1000
#     c = 0
#     c_inside_of_boundary = 0
#     c_out_of_boundary = 0

#     if debug:
#         print()

#     while True:
#         c += 1

#         if debug:
#             print(c, 'NelderMead state:', nelder_mead._state,
#                   'executing:', nelder_mead._executing_index,
#                   'evaluated_itr:', nelder_mead._evaluated_itr)

#         # a functionality of NelderMeadOptimizer::check_result()
#         # ready_params = nelder_mead.get_ready_parameters()
#         ready_params = nelder_mead._executing

#         for rp in ready_params:
#             rp['result'] = sum([pp['value'] ** 2 for pp in rp['parameters']])
#             nelder_mead.add_result_parameters(rp)

#             if debug:
#                 print('\tsum:', rp['result'])

#         # a functionality of NelderMeadOptimizer::generate_parameter()
#         searched_params = nelder_mead.search()

#         if searched_params is None:
#             if debug:
#                 print('Reached to max iteration.')
#             break

#         if len(searched_params) == 0:
#             continue

#         if debug:
#             for sp in searched_params:
#                 print('\t', sp['name'], sp['state'], sp['itr'],
#                       sp['out_of_boundary'])

#         for sp in searched_params:
#             if sp['out_of_boundary']:
#                 c_out_of_boundary += 1
#             else:
#                 c_inside_of_boundary += 1

#         if c >= c_max:
#             break

#     if debug:
#         print()
#         print('inside of boundary', c_inside_of_boundary)
#         print('out of boundary', c_out_of_boundary)

#     assert c_out_of_boundary == 0
