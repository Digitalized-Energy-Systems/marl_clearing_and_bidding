
import numpy as np
import pandapower.networks as pn
import pandapower as pp

net = pn.create_cigre_network_mv(with_der="pv_wind")

# Set the system constraints
# Define the voltage band of +-5%
# net.bus['max_vm_pu'] = 1.05
# net.bus['min_vm_pu'] = 0.95
# # Set maximum loading of lines and transformers
# net.line['max_loading_percent'] = 80
# net.trafo['max_loading_percent'] = 80

# net.load['max_p_mw'] = net.load['p_mw'] * 1.0
# net.load['min_p_mw'] = net.load['p_mw'] * 0.05
# net.load['max_q_mvar'] = net.load['max_p_mw'] * 0.3
# net.load['min_q_mvar'] = net.load['min_p_mw'] * 0.3

# net.load['p_mw'] = net.load['max_p_mw'] * 0.5
# net.load['q_mvar'] = net.load['max_p_mw'] * 0.5
# net.load['controllable'] = False

net.sgen['max_p_mw'] = net.sgen['p_mw'] * 1.0
net.sgen['min_p_mw'] = np.zeros(len(net.sgen.index))
net.sgen['max_s_mva'] = net.sgen['max_p_mw'] / 0.95  # = cos phi
net.sgen['max_q_mvar'] = net.sgen['max_s_mva']
net.sgen['min_q_mvar'] = -net.sgen['max_s_mva']

# net.sgen['p_mw'] = net.sgen['max_p_mw'] * 1
# net.sgen['q_mvar'] = net.sgen['max_p_mw'] * 1
net.sgen['controllable'] = True

for i in net.sgen.index:
    pp.create_poly_cost(net, i, 'sgen',
                        cp1_eur_per_mw=-15, cq2_eur_per_mvar2=2.5)

pp.runopp(net)

print('success!')
