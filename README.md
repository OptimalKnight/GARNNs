# Evaluating Self-Driving Car Performance with RNNs and Genetic Algorithm Optimization

This Unity project implements a genetic algorithm for self-driving cars, focusing on evaluating the performance of Recurrent Neural Networks (RNNs) and Artificial Neural Networks (ANNs) with Genetic Algorithm (GA) optimization. The project comprises the following classes:

- **GeneticAlgorithmManager.cs**: Orchestrates the genetic algorithm's workflow, managing populations through crossover, mutation, and selection of the best-performing individuals.
- **CarController.cs**: Controls the car's movement, calculating fitness based on distance traveled, speed, and sensor readings.
- **NeuralNetwork.cs**: Represents a neural network, handling initialization, mutation, crossover, and evaluation of inputs to produce outputs.
- **RecurrentNeuralNetwork.cs**: Inherits from `NeuralNetwork` and adds recurrent connections between hidden layers for more complex learning.

## Authors

- [@OptimalKnight](https://www.github.com/OptimalKnight)
- [@Gautam7702](https://www.github.com/Gautam7702)
- [@prxkhxr](https://www.github.com/prxkhxr) 

## Compilation and Build

The Unity project doesn't require manual compilation as it's developed within the Unity environment. However, ensure the following steps:

1. **Open in Unity**: Open the project folder in the Unity Editor.
2. **Unity Editor Configuration**:  Check settings like target platform, rendering settings, and other project-specific configurations within the Unity Editor.
3. **Build Settings**: Set the desired build settings (e.g., platform, scene configurations) using the Unity Editor's build settings.

## Execution

To run the genetic algorithm self-driving simulation:

1. **Launch Simulation**: Start the simulation within the Unity Editor by clicking the play button.
2. **Observation**: Observe the simulation as the genetic algorithm evolves neural networks for self-driving cars.
3. **Evaluation**: Monitor the car's behavior, fitness progression across generations, and the final evolved neural network's performance.
4. **Debugging & Analysis**: Use Unity's debugging tools, logs, or custom monitoring within the code to analyze the algorithm's behavior and neural network performance.

## Considerations

- **File Locations**: The code provided assumes specific file paths for data storage. Make sure the file writing and reading operations have the necessary permissions and file paths set correctly.
- **Experiment Configuration**: Experiment-specific data (such as number of generations, mutation rates, and population sizes) may need adjustment based on specific requirements.
- **Customization**: Modify parameters, neural network structures, and algorithmic behaviors as needed for experimentation or improvement.

For a detailed understanding of the research, experimental setup, and results, refer to the provided research paper in the repository: `Evaluating Self-Driving Car Performance with RNNs and Genetic Algorithm Optimization.pdf`.
