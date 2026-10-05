import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

M = 30
q = 2
L = 4
num_levels = 14

lambda_params = [0.0005, 0.0005]
lambda_0, lambda_1 = lambda_params

steps = 30
p_layers = 2

costs = []
latencies = []

bit_weights = np.array([10.0, 2.0, 4.0, 6.0])

P_max = np.ones(M) * (10 ** (25 / 10))

B_channel = 20e6 / 3
noise_dBm = -174 + 10 * np.log10(B_channel)
sigma_sq = 10 ** ((noise_dBm - 30) / 10)

distance = 2e2

np.random.seed(10)

channel_gain = 10 ** (np.random.uniform(-15, -8, M) / 10)
interference = np.random.uniform(0.8e10, 1.2e10, M)

def decode_power(bits, user):
    power_index = np.dot(bit_weights, bits)
    power = (P_max[user] / 13.0) * power_index
    return power

def create_user_qubo(user):
    Q = np.zeros((L, L))
    h_i = channel_gain[user]
    I_i = interference[user]
    power_weight = (P_max[user] / 13.0) * bit_weights

    rate_coefficient = B_channel * h_i / (np.log(2) * (I_i + sigma_sq))

    for ell in range(L):
        Q[ell, ell] += -lambda_0 * rate_coefficient * power_weight[ell]

    for ell in range(L):
        Q[ell, ell] += lambda_1 * power_weight[ell] ** 2

        for m in range(ell + 1, L):
            coupling = 2.0 * lambda_1 * power_weight[ell] * power_weight[m]
            Q[ell, m] += coupling / 2.0
            Q[m, ell] += coupling / 2.0

    max_value = np.max(np.abs(Q))

    if max_value > 0:
        Q = Q / max_value

    return Q

def qubo_to_ising(Q):
    n = Q.shape[0]
    h = np.zeros(n)
    J = np.zeros((n, n))

    for i in range(n):
        h[i] += -Q[i, i] / 2.0

    for i in range(n):
        for j in range(i + 1, n):
            b_ij = 2.0 * Q[i, j]
            h[i] += -b_ij / 4.0
            h[j] += -b_ij / 4.0
            J[i, j] = b_ij / 4.0
            J[j, i] = J[i, j]

    return h, J

def prepare_initial_superposition():
    for i in range(q):
        qml.Hadamard(wires=i)

def qaoa_layer(gamma, beta, h_block, J_block):
    for i in range(q):
        qml.RZ(2.0 * gamma * h_block[i], wires=i)

    if J_block[0, 1] != 0:
        qml.CNOT(wires=[0, 1])
        qml.RZ(2.0 * gamma * J_block[0, 1], wires=1)
        qml.CNOT(wires=[0, 1])

    for i in range(q):
        qml.RX(2.0 * beta, wires=i)

dev = qml.device("default.qubit", wires=2)

def make_segment_expectation_qnode(h_block, J_block):
    @qml.qnode(dev)
    def circuit(params):
        depth = len(params) // 2
        gamma = params[:depth]
        beta = params[depth:]

        prepare_initial_superposition()

        for layer in range(depth):
            qaoa_layer(gamma[layer], beta[layer], h_block, J_block)

        return (
            qml.expval(qml.PauliZ(0)),
            qml.expval(qml.PauliZ(1)),
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        )

    return circuit

def segment_energy(expvals, h_block, J_block):
    z0, z1, z01 = expvals
    return h_block[0] * z0 + h_block[1] * z1 + J_block[0, 1] * z01

def make_reuse_segments(Q):
    h, J = qubo_to_ising(Q)
    segments = []

    h_1 = np.array([h[0], h[1]])
    J_1 = np.zeros((2, 2))
    J_1[0, 1] = J[0, 1]
    J_1[1, 0] = J[1, 0]
    segments.append((h_1, J_1))

    h_2 = np.array([h[2], h[3]])
    J_2 = np.zeros((2, 2))
    J_2[0, 1] = J[2, 3]
    J_2[1, 0] = J[3, 2]
    segments.append((h_2, J_2))

    return segments

Q_list = [create_user_qubo(user) for user in range(M)]
reuse_segments = [make_reuse_segments(Q) for Q in Q_list]

print("Number of users:", M)
print("Logical binary variables:", M * L)
print("Physical qubits:", q)
print("Reuse segments:", M * (L // q))

def total_network_energy(params):
    total = 0.0

    for user in range(M):
        for h_block, J_block in reuse_segments[user]:
            circuit = make_segment_expectation_qnode(h_block, J_block)
            expvals = circuit(params)
            total += segment_energy(expvals, h_block, J_block)

    return total

optimizer = qml.GradientDescentOptimizer(stepsize=0.002)

params = np.random.uniform(
    0,
    np.pi,
    2 * p_layers,
    requires_grad=True
)

for step in range(steps):
    params = optimizer.step(total_network_energy, params)
    E_total = total_network_energy(params)

    qaoa_objective = -E_total
    costs.append(qaoa_objective)

    average_latency = distance / (qaoa_objective + 1e-12)
    latencies.append(average_latency)

    print(
        f"Iteration {step:3d} | "
        f"Latency = {average_latency:.6f}"
    )

plt.figure(figsize=(6, 4))
plt.plot(latencies, marker='o', linestyle='-', color='b')
plt.xlabel("Iteration")
plt.ylabel("Average Latency")
plt.grid(True)
plt.tight_layout()
plt.show()