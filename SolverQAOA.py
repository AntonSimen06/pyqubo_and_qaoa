import scipy
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pyqubo import Binary, Spin
from pprint import pprint
from dimod import ExactSolver
#myqlm imports
from qat.lang.AQASM import *
from qat.lang.AQASM import *
from qat.qpus import get_default_qpu

class SolverQAOA:

    """
    Atributes:
        binary_model -> a model() from pyqubo using Binary(xi) variables.
        num_variables -> Integer number of Binary() variables in the QUBO.
        num_measurements -> Integer number which means number of measurements on quantum circuit.
        depth -> Integer number representing number of QAOA layers.
    """

    def __init__(self, binary_model, num_variables, num_measurements, depth):

        self.binary_model = binary_model
        self.num_variables = num_variables
        self.num_measurements = num_measurements
        self.depth = depth

    def phase_op(self, gamma):

        """
        Args:
            gamma -> float representing a variational parameter
        Output:
            circuit -> quantum circuit that encoding the Ising hamiltonian
        """

        circuit = QRoutine()
        wires = circuit.new_wires(self.num_variables)

        ising_model = self.binary_model.to_ising()
        linear, quadratic = ising_model[0], ising_model[1]

        with circuit.compute():

            for key, value in quadratic.items():
                CNOT(wires[int(key[0][1])] , wires[int(key[1][1])])
                RZ(2*gamma*value)(wires[int(key[1][1])])
                CNOT(wires[int(key[0][1])] , wires[int(key[1][1])])

            for key, value in linear.items():
                RZ(2*gamma*value)(wires[int(key[1])])

        return circuit

    def mixer_op(self, beta):

        """
        Args:
            beta -> float representing a variational parameter
        Output:
            circuit -> quantum circuit for mixer operator
        """

        circuit = QRoutine()
        wires = circuit.new_wires(self.num_variables)

        with circuit.compute():

            for wire in wires:
                RX(2*beta)(wire)

        return circuit

    def full_circuit_measurements(self, params):

        """
        Args:
            params -> array representing all variational parameters
        Outputs:
            circuit -> full circuit for all layers
            meas -> dictionary with eigenstates (strings in the keys) and probabilities (values)
        """

        #create program
        circuit = Program()
        qbits = circuit.qalloc(self.num_variables)

        for qubit in qbits:
            H(qubit)

        for layer in range(self.depth):

            self.phase_op(params[2*layer])(qbits)
            self.mixer_op(params[2*layer+1])(qbits)

        qc = circuit.to_circ()
        job = qc.to_job(nbshots = self.num_measurements)
        result = get_default_qpu().submit(job)

        meas = {}
        for sample in result:
            #sample._state returns quantum state in the decimal basis
            meas[sample.state] = sample.probability
        
        return qc, meas

    def cost_function(self, eigenstate):

        """
        Args:
            eigenstate -> binary string 
        Output:
            cost -> cost associated with binary string regarding the QUBO.
        """
        
        #Ising model to qubo
        qubo = self.binary_model.to_qubo()
        #evalueate cost function
        cost = 0
        for key, value in qubo[0].items():
            cost += value * int(eigenstate[int(key[0][1])]) * int(eigenstate[int(key[1][1])])
        
        return cost

    def expected_value(self, params):

        """
        Args:
            params -> array with all variational parameters
        Output:
            exp_value -> float that is the expected value from measurements
        """

        circ, measurements = self.full_circuit_measurements(params)

        # measuring expected values <psi|H|psi>
        exp_value = 0
        for key, value in measurements.items():
            exp_value += value * self.cost_function(str(key)[1:-1])

        #print("Expected value: ", exp_value)        
        conv.append(exp_value)

        return exp_value

    def run(self):

        #iteration=1
        convergence = []
        def callback(variational_parameters):
            #global iteration
            convergence.append(self.expected_value(variational_parameters))
            print(" \t Expected value from measurements: ",  self.expected_value(variational_parameters), " \t Variational parameters: ", variational_parameters)
            #iteration = iteration + 1

        #x0 = np.random.uniform(0, np.pi, self.depth*self.num_variables
        res = scipy.optimize.minimize(self.expected_value, x0=np.ones(self.depth*self.num_variables), 
                                        method = 'COBYLA', callback=callback,
                                        options={'maxiter': 200, 'ftol': 1e-06, 'iprint': 1, 'disp': True, 
                                        'eps': 1.4901161193847656e-08, 'finite_diff_rel_step': None})


        result = {
            
            "best_params": res['x'],
            "energies": convergence,
            "final_circuit": self.full_circuit_measurements(res['x'])[0],
            "final_measurements": self.full_circuit_measurements(res['x'])[1]

        }

        #getting best solution from final measurements
        eigstates = []; probs = []
        for key, values in result["final_measurements"].items():
            eigstates.append(key)
            probs.append(values)
        
        best_result = eigstates[probs.index(max(probs))]
        print("\n \n QAOA solution: ", best_result)
        result["best_solution"] = best_result

        #print("Approximation ratio: ", self.cost_function(str(best_result)[1:-1])/2)

        bqm = self.binary_model.to_bqm()
        sampleset = ExactSolver().sample(bqm)
        decoded_samples = self.binary_model.decode_sampleset(sampleset)
        best_sample = min(decoded_samples, key=lambda s: s.energy)
        #print(best_sample.energy)
        print("Pyqubo solution: ", best_sample.sample)

        return result

# COBYLA does not support callback, then we create conv to store energies
# testing with a simple QUBO
conv = []
def main():
    
    h =( Binary("x0")*Binary("x2") - 2*Binary("x1")*Binary("x0") + 5*Binary("x3")*Binary("x1") )
    model = h.compile()
    qaoa_myqlm = SolverQAOA(binary_model = model, num_variables = 4, num_measurements=1000, depth=4)
    res = qaoa_myqlm.run()


if __name__ == "__main__":
    main()
