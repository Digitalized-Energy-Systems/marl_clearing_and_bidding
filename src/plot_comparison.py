
import matplotlib.pyplot as plt

n_agents = [10, 20, 30, 40]
regret_maddpg_10000 = [1.2665969981697789,
                       2.557193847462922,
                       2.7894554413416657,
                       2.8689075113063316]
regret_mmaddpg_10000 = [1.2682381220591337,
                        0.9805369902566451,
                        1.103446876349354,
                        2.2157721062070026]
regret_maddpg_30000 = [0.7490424130764557,
                       0.8022675823972255,
                       0.8383680295397797,
                       0.7874453048072257]
regret_mmaddpg_30000 = [1.1288130414518809,
                        0.9595709514191901,
                        0.9444775012458528,
                        1.0334768663563518]

plt.plot(n_agents, regret_maddpg_10000, label='MADDPG 10k')
plt.plot(n_agents, regret_mmaddpg_10000, label='M-MADDPG 10k')
plt.plot(n_agents, regret_maddpg_30000, label='MADDPG 30k')
plt.plot(n_agents, regret_mmaddpg_30000, label='M-MADDPG 30k')

plt.legend(loc='upper left')
plt.xlabel('n agents')
plt.ylabel('total regret')

plt.show()
