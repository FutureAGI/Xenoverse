import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lti, StateSpace, step, impulse, bode
from itertools import product

class MIMOPolynomialTransferFunctionSimulator:
    """
    Simulator for multi-input multi-output (MIMO) systems based on polynomial transfer functions
    
    Parameters:
    num_mats: Numerator polynomial coefficient matrices (from highest to lowest degree)
    den_mats: Denominator polynomial coefficient matrices (from highest to lowest degree)
    num_inputs: Number of inputs
    num_outputs: Number of outputs
    time_step: Simulation time step
    max_time: Maximum simulation time
    """
    
    def __init__(self, num_mats, den_mats, num_inputs, num_outputs, 
                 time_step=0.01, max_time=10.0):
        self.num_mats = num_mats
        self.den_mats = den_mats
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.time_step = time_step
        self.max_time = max_time
        
        # Create time vector
        self.time_vector = np.arange(0, self.max_time + self.time_step, self.time_step)
        
        # Validate dimensions of numerator and denominator matrices
        if len(num_mats) != num_outputs * num_inputs:
            raise ValueError(f"Number of numerator matrices should be {num_outputs * num_inputs}, actual {len(num_mats)}")
        
        if len(den_mats) != num_outputs * num_inputs:
            raise ValueError(f"Number of denominator matrices should be {num_outputs * num_inputs}, actual {len(den_mats)}")
        
        # Create transfer function matrix
        self.tf_matrix = []
        for i in range(num_outputs):
            row = []
            for j in range(num_inputs):
                idx = i * num_inputs + j
                row.append(lti(num_mats[idx], den_mats[idx]))
            self.tf_matrix.append(row)
        
        # Create random input association matrix
        self.input_matrix = np.random.rand(num_inputs, num_inputs)
    
    def evaluate(self, input_signals):
        """
        Simulate the system for given input signals
        
        Parameters:
        input_signals: List of input signal functions, each function takes time as input and returns value
        
        Returns:
        Time vector and output signals
        """
        if len(input_signals) != self.num_inputs:
            raise ValueError(f"Number of input signals should be {self.num_inputs}, actual {len(input_signals)}")
        
        # Calculate input signals
        u = np.array([[input_signals[i](t) for i in range(self.num_inputs)] 
                      for t in self.time_vector])
        
        # Apply random input matrix
        u = np.dot(u, self.input_matrix)
        
        # Initialize output
        y = np.zeros((len(self.time_vector), self.num_outputs))
        
        # Simulate each transfer function
        for o in range(self.num_outputs):
            for i in range(self.num_inputs):
                # Get current input signal
                current_input = u[:, i]
                
                # Get current transfer function
                tf = self.tf_matrix[o][i]
                
                # Calculate output
                t_out, y_out = step(tf, T=self.time_vector)
                
                # If there is output, add to total output
                if len(y_out) > 0:
                    y[:, o] += y_out
        
        return self.time_vector, y
    
    def step_response(self):
        """Calculate step response"""
        # Initialize output
        y = np.zeros((len(self.time_vector), self.num_outputs))
        
        # Simulate step response for each transfer function
        for o in range(self.num_outputs):
            for i in range(self.num_inputs):
                # Get current transfer function
                tf = self.tf_matrix[o][i]
                
                # Calculate step response
                t_out, y_out = step(tf, T=self.time_vector)
                
                # If there is output, add to total output
                if len(y_out) > 0:
                    y[:, o] += y_out
        
        return self.time_vector, y
    
    def impulse_response(self):
        """Calculate impulse response"""
        # Initialize output
        y = np.zeros((len(self.time_vector), self.num_outputs))
        
        # Simulate impulse response for each transfer function
        for o in range(self.num_outputs):
            for i in range(self.num_inputs):
                # Get current transfer function
                tf = self.tf_matrix[o][i]
                
                # Calculate impulse response
                t_out, y_out = impulse(tf, T=self.time_vector)
                
                # If there is output, add to total output
                if len(y_out) > 0:
                    y[:, o] += y_out
        
        return self.time_vector, y
    
    def frequency_response(self):
        """Calculate frequency response"""
        w = np.logspace(-2, 3, 500)
        mag = np.zeros((len(w), self.num_outputs, self.num_inputs))
        phase = np.zeros((len(w), self.num_outputs, self.num_inputs))
        
        # Simulate frequency response for each transfer function
        for o in range(self.num_outputs):
            for i in range(self.num_inputs):
                # Get current transfer function
                tf = self.tf_matrix[o][i]
                
                # Calculate frequency response
                w_i, mag_i, phase_i = bode(tf, w=w)
                
                mag[:, o, i] = mag_i
                phase[:, o, i] = phase_i
        
        return w, mag, phase
    
    def plot_time_response(self, input_signals=None, step_response=True, impulse_response=True):
        """
        Plot time response
        
        Parameters:
        input_signals: List of input signal functions, each function takes time as input and returns value
        step_response: Whether to plot step response
        impulse_response: Whether to plot impulse response
        """
        plt.figure(figsize=(12, 8))
        
        if input_signals:
            t_out, y_out = self.evaluate(input_signals)
            
            for o in range(self.num_outputs):
                plt.plot(t_out, y_out[:, o], label=f'Output {o+1}')
        
        if step_response:
            t_step, y_step = self.step_response()
            for o in range(self.num_outputs):
                plt.plot(t_step, y_step[:, o], '--', label=f'Step Response {o+1}')
        
        if impulse_response:
            t_impulse, y_impulse = self.impulse_response()
            for o in range(self.num_outputs):
                plt.plot(t_impulse, y_impulse[:, o], '-.', label=f'Impulse Response {o+1}')
        
        plt.title('System Time Response')
        plt.xlabel('Time (s)')
        plt.ylabel('Output')
        plt.legend()
        plt.grid()
        plt.show()
    
    def plot_frequency_response(self):
        """Plot frequency response"""
        w, mag, phase = self.frequency_response()
        
        fig, axs = plt.subplots(self.num_outputs, self.num_inputs, figsize=(12, 6 * self.num_outputs))
        
        # Adjust title
        fig.suptitle('Frequency Response')
        
        for o, i in product(range(self.num_outputs), range(self.num_inputs)):
            # Magnitude response
            if self.num_outputs > 1 or self.num_inputs > 1:
                ax_mag = axs[o, i]
                ax_phase = axs[o, i]
            else:
                ax_mag = axs[0]
                ax_phase = axs[1]
            
            ax_mag.semilogx(w, mag[:, o, i])
            ax_mag.set_title(f'Magnitude Response (Output {o+1}, Input {i+1})')
            ax_mag.set_ylabel('Magnitude (dB)')
            ax_mag.grid(True)
            
            # Phase response
            ax_phase.twinx()
            ax_phase.semilogx(w, phase[:, o, i], 'r')
            ax_phase.set_xlabel('Frequency (rad/s)')
            ax_phase.set_ylabel('Phase (degrees)', color='r')
            ax_phase.grid(True)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    
    @staticmethod
    def create_input_signal(amplitude=1.0, frequency=1.0, phase=0.0, type='sin'):
        """
        Create input signal function
        
        Parameters:
        amplitude: Signal amplitude
        frequency: Signal frequency
        phase: Signal phase
        type: Signal type ('sin', 'cos', 'step', 'square')
        
        Returns:
        Input signal function
        """
        if type == 'sin':
            return lambda t: amplitude * np.sin(2 * np.pi * frequency * t + phase)
        elif type == 'cos':
            return lambda t: amplitude * np.cos(2 * np.pi * frequency * t + phase)
        elif type == 'step':
            return lambda t: amplitude if t >= 0 else 0
        elif type == 'square':
            return lambda t: amplitude * (1 if np.sin(2 * np.pi * frequency * t + phase) >= 0 else -1)
        else:
            raise ValueError("Unsupported signal type")
    
    def __str__(self):
        """String representation of transfer function matrix"""
        output = "Transfer Function Matrix:\n"
        for o in range(self.num_outputs):
            output += f"Output {o+1}:\n"
            for i in range(self.num_inputs):
                num = " + ".join([f"{c} s^{i}" for i, c in enumerate(reversed(self.num_mats[o * self.num_inputs + i]))])
                den = " + ".join([f"{c} s^{i}" for i, c in enumerate(reversed(self.den_mats[o * self.num_inputs + i]))])
                output += f"  Input {i+1}: ({num}) / ({den})\n"
        return output


# Example usage
if __name__ == "__main__":
    # Define system parameters
    num_inputs = 2  # Number of inputs
    num_outputs = 2  # Number of outputs
    
    # Define numerator and denominator polynomial coefficients for each transfer function
    num_mats = [
        [2, 1], [1, 3],  # Numerators for output 1
        [3, 1], [2, 2]   # Numerators for output 2
    ]
    
    den_mats = [
        [1, 3, 2], [1, 2, 1],  # Denominators for output 1
        [1, 4, 3], [1, 1, 1]   # Denominators for output 2
    ]
    
    simulator = MIMOPolynomialTransferFunctionSimulator(
        num_mats, den_mats, num_inputs, num_outputs)
    
    print(simulator)
    
    # Define input signals
    input_signals = [
        simulator.create_input_signal(amplitude=1.0, frequency=1.0, phase=0.0, type='sin'),
        simulator.create_input_signal(amplitude=1.0, frequency=0.5, phase=np.pi/2, type='cos')
    ]
    
    # Plot frequency response
    simulator.plot_frequency_response()
    
    # Plot time response
    simulator.plot_time_response(input_signals)