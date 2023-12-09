using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Reflection;
using UnityEngine;
using UnityEngine.Assertions;
using Random = UnityEngine.Random;

public class NeuralNetwork : MonoBehaviour
{

    public Matrix<float> inputLayer, outputLayer;
    public List<Matrix<float>> hiddenLayers, weights;
    public List<float> biases;
    public float fitness;

    public void Initialize(int inputLayerCount, int outputLayerCount, int hiddenLayerCount, int hiddenNeuronCount)
    {
        inputLayer = Matrix<float>.Build.Dense(1, inputLayerCount);
        outputLayer = Matrix<float>.Build.Dense(1, outputLayerCount);
        hiddenLayers = new();
        weights = new();
        biases = new();

        for (int i = 0; i <= hiddenLayerCount; i++)
        {
            hiddenLayers.Add(Matrix<float>.Build.Dense(1, hiddenNeuronCount));
            biases.Add(Random.Range(-1f, 1f));

            if (i == 0)
            {
                weights.Add(Matrix<float>.Build.Dense(inputLayer.ColumnCount, hiddenNeuronCount));
                continue;
            }

            weights.Add(Matrix<float>.Build.Dense(hiddenNeuronCount, hiddenNeuronCount));
        }

        weights.Add(Matrix<float>.Build.Dense(hiddenNeuronCount, outputLayer.ColumnCount));
        biases.Add(Random.Range(-1f, 1f));
        AssignWeights();
    }

    private void AssignWeights()
    {
        for (int i = 0; i < weights.Count; i++)
        {
            for (int x = 0; x < weights[i].RowCount; x++)
            {
                for (int y = 0; y < weights[i].ColumnCount; y++)
                {
                    weights[i][x, y] = Random.Range(-1f, 1f);
                }
            }
        }
    }

    public NeuralNetwork Copy()
    {
        NeuralNetwork neuralNetwork = new();
        neuralNetwork.Initialize(inputLayer.ColumnCount, outputLayer.ColumnCount, hiddenLayers.Count - 1, hiddenLayers[0].ColumnCount);
        for (int i = 0; i < weights.Count; i++)
        {
            for (int x = 0; x < weights[i].RowCount; x++)
            {
                for (int y = 0; y < weights[i].ColumnCount; y++)
                {
                    neuralNetwork.weights[i][x, y] = weights[i][x, y];
                }
            }
        }

        for (int i = 0; i < biases.Count; i++)
        {
            neuralNetwork.biases[i] = biases[i];
        }

        return neuralNetwork;
    }

    public (float, float) RunNetwork(List<float> input)
    {
        for (int i = 0; i < input.Count; i++)
        {
            inputLayer[0, i] = input[i];
        }

        inputLayer = inputLayer.PointwiseTanh();
        hiddenLayers[0] = ((inputLayer * weights[0]) + biases[0]).PointwiseTanh();

        for (int i = 1; i < hiddenLayers.Count; i++)
        {
            hiddenLayers[i] = ((hiddenLayers[i - 1] * weights[i]) + biases[i]).PointwiseTanh();
        }

        outputLayer = ((hiddenLayers[^1] * weights[^1]) + biases[^1]).PointwiseTanh();
        return (Sigmoid(outputLayer[0, 0]), (float)Math.Tanh(outputLayer[0, 1]));
    }

    protected float Sigmoid(float x)
    {
        return (1 / (1 + Mathf.Exp(-x)));
    }

    public static (NeuralNetwork, NeuralNetwork) Crossover(NeuralNetwork parent1, NeuralNetwork parent2)
    {
        NeuralNetwork child1 = new();
        NeuralNetwork child2 = new();

        child1.Initialize(parent1.inputLayer.ColumnCount, parent1.outputLayer.ColumnCount, parent1.hiddenLayers.Count - 1, parent1.hiddenLayers[0].ColumnCount);
        child2.Initialize(parent2.inputLayer.ColumnCount, parent2.outputLayer.ColumnCount, parent2.hiddenLayers.Count - 1, parent2.hiddenLayers[0].ColumnCount);

        for (int i = 0; i < child1.weights.Count; i++)
        {
            for (int x = 0; x < child2.weights[i].RowCount; x++)
            {
                for (int y = 0; y < child1.weights[i].ColumnCount; y++)
                {
                    child1.weights[i][x, y] = parent1.weights[i][x, y];
                    child2.weights[i][x, y] = parent2.weights[i][x, y];

                    if (Random.Range(0.0f, 1.0f) < 0.5f)
                    {
                        (child2.weights[i], child1.weights) = (child1.weights[i], child2.weights);
                    }
                }
            }
        }

        for (int i = 0; i < child1.biases.Count; i++)
        {
            child1.biases[i] = parent1.biases[i];
            child2.biases[i] = parent2.biases[i];

            if (Random.Range(0.0f, 1.0f) < 0.5f)
            {
                (child2.biases[i], child1.biases) = (child1.biases[i], child2.biases);
            }
        }

        return (child1, child2);
    }

    public void Mutate(float mutationRate)
    {
        for (int i = 0; i < weights.Count; i++)
        {
            if (Random.Range(0.0f, 1.0f) < mutationRate)
            {
                int randomPoints = Random.Range(1, (weights[i].RowCount * weights[i].ColumnCount) / 7);

                for (int j = 0; j < randomPoints; j++)
                {
                    int randomColumn = Random.Range(0, weights[i].ColumnCount);
                    int randomRow = Random.Range(0, weights[i].RowCount);

                    weights[i][randomRow, randomColumn] = Mathf.Clamp(weights[i][randomRow, randomColumn] + Random.Range(-1f, 1f), -1f, 1f);
                }
            }
        }

        for (int i = 0; i < biases.Count; i++)
        {
            if (Random.Range(0.0f, 1.0f) < mutationRate)
            {
                biases[i] = Mathf.Clamp(biases[i] + Random.Range(-1f, 1f), -1f, 1f);
            }
        }
    }

}
