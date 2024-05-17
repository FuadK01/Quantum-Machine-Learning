import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile, assemble
from qiskit.tools.visualization import circuit_drawer, plot_histogram
from qiskit.quantum_info import state_fidelity
from qiskit_aer import AerSimulator
import Benchmarking
from autograd.numpy.numpy_boxes import ArrayBox
from pennylane.numpy import tensor as qml_tensor


""" Deprecated Functions """
def old_conv_layer1(U, qc, params, qregister):
    U(qc, params, [qregister[0], qregister[7]])
    for i in range(0, 8, 2):
        U(qc, params, [qregister[i], qregister[i + 1]])
    for i in range(1, 7, 2):
        U(qc, params, [qregister[i], qregister[i + 1]])

def old_conv_layer2(U, qc, params, qregister):
    U(qc, params, [qregister[0], qregister[6]])
    U(qc, params, [qregister[0], qregister[2]])
    U(qc, params, [qregister[4], qregister[6]])
    U(qc, params, [qregister[2], qregister[4]])

def old_pooling_layer1(V, qc, params, qregister):
    for i in range(0, 8, 2):
        V(qc, params, [qregister[i+1], qregister[i]])

def old_pooling_layer2(V, qc, params, qregister):
    V(qc, params, [qregister[2], qregister[0]])
    V(qc, params, [qregister[6], qregister[4]])

def pooling_layer3(V, qc, params, qregister):
    V(qc, params, [qregister[0], qregister[4]])

def old_QCNN_structure(U, qc, params, U_params, qregister):
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]
    parama = params[3 * U_params: 4 * U_params]
    paramb = params[4 * U_params: 5 * U_params]
    paramc = params[5 * U_params: 6 * U_params]
    param4 = params[6 * U_params: 6 * U_params + 2]
    param5 = params[6 * U_params + 2: 6 * U_params + 4]
    param6 = params[6 * U_params + 4: 6 * U_params + 19]

    old_conv_layer1(U, qc, param1, qregister)
    old_conv_layer1(U, qc, parama, qregister)
    old_pooling_layer1(Pooling_ansatz1, qc, param4, qregister)
    old_conv_layer2(U, qc, param2, qregister)
    old_conv_layer2(U, qc, paramb, qregister)
    old_pooling_layer2(Pooling_ansatz1, qc, param5, qregister)
    conv_layer3(U, qc, param3, qregister)
    conv_layer3(U, qc, paramc, qregister)
    U_SU4(qc, param6, [qregister[0], qregister[4]])




# Very unexpressive - CC1
def U_TTN(qc, params, qregister):  # 2 params
    qc.ry(params[0], qregister[0])
    qc.ry(params[1], qregister[1])
    qc.cx(qregister[0], qregister[1])

# Very expressive - CC7
def U_5(qc, params, qregister):  # 10 params
    qc.rx(params[0], qregister[0])
    qc.rx(params[1], qregister[1])
    qc.rz(params[2], qregister[0])
    qc.rz(params[3], qregister[1])
    qc.crz(params[4], qregister[1], qregister[0])
    qc.crz(params[5], qregister[0], qregister[1])
    qc.rx(params[6], qregister[0])
    qc.rx(params[7], qregister[1])
    qc.rz(params[8], qregister[0])
    qc.rz(params[9], qregister[1])

# Very expressive - CC8
def U_6(qc, params, qregister):  # 10 params
    qc.rx(params[0], qregister[0])
    qc.rx(params[1], qregister[1])
    qc.rz(params[2], qregister[0])
    qc.rz(params[3], qregister[1])
    qc.crx(params[4], qregister[1], qregister[0])
    qc.crx(params[5], qregister[0], qregister[1])
    qc.rx(params[6], qregister[0])
    qc.rx(params[7], qregister[1])
    qc.rz(params[8], qregister[0])
    qc.rz(params[9], qregister[1])

# Somewhat expressive - CC2
def U_9(qc, params, qregister):  # 2 params
    qc.h(qregister[0])
    qc.h(qregister[1])
    qc.cz(qregister[0], qregister[1])
    qc.rx(params[0], qregister[0])
    qc.rx(params[1], qregister[1])

# Somewhat expressive - CC4
def U_13(qc, params, qregister):  # 6 params
    qc.ry(params[0], qregister[0])
    qc.ry(params[1], qregister[1])
    qc.crz(params[2], qregister[1], qregister[0])
    qc.ry(params[3], qregister[0])
    qc.ry(params[4], qregister[1])
    qc.crz(params[5], qregister[0], qregister[1])

# Somewhat expressive - CC5
def U_14(qc, params, qregister):  # 6 params
    qc.ry(params[0], qregister[0])
    qc.ry(params[1], qregister[1])
    qc.crx(params[2], qregister[1], qregister[0])
    qc.ry(params[3], qregister[0])
    qc.ry(params[4], qregister[1])
    qc.crx(params[5], qregister[0], qregister[1])

# Somewhat expressive - CC3
def U_15(qc, params, qregister):  # 4 params
    qc.ry(params[0], qregister[0])
    qc.ry(params[1], qregister[1])
    qc.cx(qregister[1], qregister[0])
    qc.ry(params[2], qregister[0])
    qc.ry(params[3], qregister[1])
    qc.cx(qregister[0], qregister[1])

# Somewhat expressive - CC6
def U_SO4(qc, params, qregister):  # 6 params
    qc.ry(params[0], qregister[0])
    qc.ry(params[1], qregister[1])
    qc.cx(qregister[0], qregister[1])
    qc.ry(params[2], qregister[0])
    qc.ry(params[3], qregister[1])
    qc.cx(qregister[0], qregister[1])
    qc.ry(params[4], qregister[0])
    qc.ry(params[5], qregister[1])

# Very expressive - CC9
def U_SU4(qc, params, qregister): # 15 params
    qc.u(params[0], params[1], params[2], qregister[0])
    qc.u(params[3], params[4], params[5], qregister[1])
    qc.cx(qregister[0], qregister[1])
    qc.ry(params[6], qregister[0])
    qc.rz(params[7], qregister[1])
    qc.cx(qregister[1], qregister[0])
    qc.ry(params[8], qregister[0])
    qc.cx(qregister[0], qregister[1])
    qc.u(params[9], params[10], params[11], qregister[0])
    qc.u(params[12], params[13], params[14], qregister[1])

# Pooling Layers
def Pooling_ansatz1(qc, params, qregister): #2 params
    qc.crz(params[0], qregister[0], qregister[1])
    qc.x(qregister[0])
    qc.crx(params[1], qregister[0], qregister[1])

def Pooling_ansatz2(qc, qregister):  # 0 params
    qc.crz(qregister[0], qregister[1])

# Quantum Circuits for Convolutional layers

def conv_layer1(U, qc, params, U_num_params, qregister):
    U(qc, params[:U_num_params], [qregister[0], qregister[7]])
    param_counter = 1
    for i in range(0, 8, 2):
        U(qc, params[U_num_params * param_counter:U_num_params * (param_counter + 1)], [qregister[i], qregister[i + 1]])
        param_counter += 1
    for i in range(1, 7, 2):
        U(qc, params[U_num_params * param_counter:U_num_params * (param_counter + 1)], [qregister[i], qregister[i + 1]])
        param_counter += 1

def conv_layer2(U, qc, params, U_num_params, qregister):
    U(qc, params[:U_num_params], [qregister[0], qregister[6]])
    U(qc, params[U_num_params: U_num_params * 2], [qregister[0], qregister[2]])
    U(qc, params[U_num_params*2:U_num_params*3], [qregister[4], qregister[6]])
    U(qc, params[U_num_params*3:U_num_params*4], [qregister[2], qregister[4]])

def conv_layer3(U, qc, params, qregister):
    U(qc, params, [qregister[0], qregister[4]])

def Dense_Layer(qc, params, qregister):
    qc.crx(params[0], qregister[0], qregister[4])
    qc.crz(params[1], qregister[4], qregister[0])
    #qc.ry(params[2], qregister[0])
    #qc.ry(params[3], qregister[4])

# Quantum Circuits for Pooling layers
def pooling_layer1(V, qc, params, qregister):
    param_counter = 0
    for i in range(0, 8, 2):
        V(qc, params[2 * param_counter:2 * (param_counter + 1)], [qregister[i+1], qregister[i]])
        param_counter += 1

def pooling_layer2(V, qc, params, qregister):
    V(qc, params[:2], [qregister[2], qregister[0]])
    V(qc, params[2:], [qregister[6], qregister[4]])

def QCNN_structure(U, qc, params, U_params, qregister):
    param1 = params[0:U_params * 8]
    param2 = params[U_params * 8: 12 * U_params]
    param3 = params[12 * U_params: 13 * U_params]
    param4 = params[13 * U_params: 13 * U_params + 8]
    param5 = params[13 * U_params + 8: 13 * U_params + 12]
    param6 = params[13 * U_params + 12: 13 * U_params + 18]

    conv_layer1(U, qc, param1, U_params, qregister)
    pooling_layer1(Pooling_ansatz1, qc, param4, qregister)
    conv_layer2(U, qc, param2, U_params, qregister)
    pooling_layer2(Pooling_ansatz1, qc, param5, qregister)
    conv_layer3(U, qc, param3, qregister)
    U_SO4(qc, param6, [qregister[0], qregister[4]])

def QCNN_structure_without_pooling(U, qc, params, U_params, qregister):
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]

    conv_layer1(U, qc, param1, qregister)
    conv_layer2(U, qc, param2, qregister)
    conv_layer3(U, qc, param3, qregister)

def QCNN_1D_circuit(U, qc, params, U_params, qregister):
    param1 = params[0: U_params]
    param2 = params[U_params: 2*U_params]
    param3 = params[2*U_params: 3*U_params]

    for i in range(0, 8, 2):
        U(qc, param1, [qregister[i], qregister[i + 1]])
    for i in range(1, 7, 2):
        U(qc, param1, [qregister[i], qregister[i + 1]])
    
    U(qc, param2, [qregister[2], qregister[3]])
    U(qc, param2, [qregister[4], qregister[5]])
    U(qc, param3, [qregister[3], qregister[4]])


def QCNN(X, params, U, U_params):
    
    """
    Pass data through QCNN

    Parameters
    ----------
    X : float
        The amplitude embedded data
    params : list, float
        Tunable parameters of entire quantum circuits
    U : string
        Unitary Ansatz used for circuit
    U_params : int
        Totaly number of parameters required for entire quantum circuit

    Returns
    -------
    list
        Probability for each dual-qubit measurment outcome

    """

    # Data type conversion
    if isinstance(params, ArrayBox):
        params = [float(param._value) for param in params]
    elif isinstance(params, qml_tensor):
        params = np.array(params)
    
    # Ensuring data is normalised
    norm_factor = np.sqrt(np.sum(np.abs(X) ** 2))
    X = X / norm_factor

    # Iniatilising quantum circuit
    qregister = QuantumRegister(8, name='q')
    cregister = ClassicalRegister(2, name='c')
    simulator = AerSimulator()
    qc = QuantumCircuit(qregister, cregister)

    # Initalising Data
    qc.initialize(X, range(8))

    # Quantum Convolutional Neural Network
    if U == 'U_TTN':
        QCNN_structure(U_TTN, qc, params, U_params, qregister)
    elif U == 'U_5':
        QCNN_structure(U_5, qc, params, U_params, qregister)
    elif U == 'U_6':
        QCNN_structure(U_6, qc, params, U_params, qregister)
    elif U == 'U_9':
        QCNN_structure(U_9, qc, params, U_params, qregister)
    elif U == 'U_13':
        QCNN_structure(U_13, qc, params, U_params, qregister)
    elif U == 'U_14':
        QCNN_structure(U_14, qc, params, U_params, qregister)
    elif U == 'U_15':
        QCNN_structure(U_15, qc, params, U_params, qregister)
    elif U == 'U_SO4':
        QCNN_structure(U_SO4, qc, params, U_params, qregister)
    elif U == 'U_SU4':
        QCNN_structure(U_SU4, qc, params, U_params, qregister)
    elif U == 'U_SU4_no_pooling':
        QCNN_structure_without_pooling(U_SU4, qc, params, U_params, qregister)
    elif U == 'U_SU4_1D':
        QCNN_1D_circuit(U_SU4, qc, params, U_params, qregister)
    elif U == 'U_9_1D':
        QCNN_1D_circuit(U_9, qc, params, U_params, qregister)
    else:
        print("Invalid Unitary Ansatz")
        return False

    qc.measure([0, 4], [0, 1])

    # Transpiling qcircuit
    tqc = transpile(qc, simulator)

    # Assemble the transpiled circuit for the simulator
    tqc_job = assemble(tqc)

    # Simulate the circuit
    result = simulator.run(tqc_job).result()

    # Get the counts from the result
    counts = result.get_counts(qc)

    #print("Measurement Results: ", counts)
    # Calculate the probability distribution
    prob_dist = np.zeros(4)
    for key, value in counts.items():
        index = int(key, 2)
        prob_dist[index] = value / 1000

    return prob_dist

if __name__ == "__main__":
    total_params = 200
    #params = np.random.randn(total_params) # Randomly initialises circuit parameters
    params = np.zeros(total_params)
    # Iniatilising quantum circuit
    qregister = QuantumRegister(8, name='q')
    cregister = ClassicalRegister(2, name='c')
    simulator = AerSimulator()
    qc = QuantumCircuit(qregister, cregister)

    QCNN_structure(U_SU4, qc, params, 15, qregister)

    qc.measure([0, 4], [0, 1])

    qc.draw("mpl")
    plt.show()
