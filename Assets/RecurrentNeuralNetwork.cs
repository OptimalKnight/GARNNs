using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using UnityEngine;

using Random = UnityEngine.Random;

public class RecurrentNeuralNetwork : NeuralNetwork
{
    
    public List<Matrix<float>> recurrentWeights = new();
    public List<float> recurrentBiases = new();

    public new void Initialize(int inputLayerCount, int outputLayerCount, int hiddenLayerCount, int hiddenNeuronCount)
    {
        base.Initialize(inputLayerCount, outputLayerCount, hiddenLayerCount, hiddenNeuronCount);
        for (int i = 0; i <= hiddenLayerCount; i++)
        {
            recurrentBiases.Add(Random.Range(-0.5f, 0.5f));

            if (i == 0)
            {
                recurrentWeights.Add(Matrix<float>.Build.Dense(hiddenNeuronCount, hiddenNeuronCount));
                continue;
            }

            recurrentWeights.Add(Matrix<float>.Build.Dense(hiddenNeuronCount, hiddenNeuronCount));
        }

        recurrentWeights.Add(Matrix<float>.Build.Dense(hiddenNeuronCount, hiddenNeuronCount));
        recurrentBiases.Add(Random.Range(-0.5f, 0.5f));
        AssignWeights();
    }

    private void AssignWeights()
    {
        for (int i = 0; i < recurrentWeights.Count; i++)
        {
            for (int x = 0; x < recurrentWeights[i].RowCount; x++)
            {
                for (int y = 0; y < recurrentWeights[i].ColumnCount; y++)
                {
                    recurrentWeights[i][x, y] = Random.Range(-0.5f, 0.5f);
                }
            }
        }
    }

    public new RecurrentNeuralNetwork Copy()
    {
        RecurrentNeuralNetwork recurrentNeuralNetwork = new();
        recurrentNeuralNetwork.Initialize(inputLayer.ColumnCount, outputLayer.ColumnCount, hiddenLayers.Count - 1, hiddenLayers[0].ColumnCount);

        for (int i = 0; i < recurrentWeights.Count; i++)
        {
            for (int x = 0; x < recurrentWeights[i].RowCount; x++)
            {
                for (int y = 0; y < recurrentWeights[i].ColumnCount; y++)
                {
                    recurrentNeuralNetwork.recurrentWeights[i][x, y] = recurrentWeights[i][x, y];
                }
            }
        }

        for (int i = 0; i < recurrentBiases.Count; i++)
        {
            recurrentNeuralNetwork.biases[i] = biases[i];
        }

        return recurrentNeuralNetwork;
    }

    public new (float, float) RunNetwork(List<float> input)
    {
        for (int i = 0; i < input.Count; i++)
        {
            inputLayer[0, i] = input[i];
        }
        inputLayer = inputLayer.PointwiseTanh();

        // Recurrent connection
        hiddenLayers[0] = ((hiddenLayers[0] * recurrentWeights[0]) + recurrentBiases[0]).PointwiseTanh();
        hiddenLayers[0] += (inputLayer * weights[0]) + biases[0];
        hiddenLayers[0] = hiddenLayers[0].PointwiseTanh();

        for (int i = 1; i < hiddenLayers.Count; i++)
        {
            hiddenLayers[i] = ((hiddenLayers[i] * recurrentWeights[i]) + recurrentBiases[i]).PointwiseTanh();
            hiddenLayers[i] += (hiddenLayers[i - 1] * weights[i]) + biases[i];
            hiddenLayers[i] = hiddenLayers[i].PointwiseTanh();
        }

        outputLayer = ((hiddenLayers[^1] * weights[^1]) + biases[^1]).PointwiseTanh();
        return (base.Sigmoid(outputLayer[0, 0]), (float)Math.Tanh(outputLayer[0, 1]));
    }

    public static (RecurrentNeuralNetwork, RecurrentNeuralNetwork) Crossover(RecurrentNeuralNetwork parent1, RecurrentNeuralNetwork parent2)
    {
        RecurrentNeuralNetwork child1 = new();
        RecurrentNeuralNetwork child2 = new();

        child1.Initialize(parent1.inputLayer.ColumnCount, parent1.outputLayer.ColumnCount, parent1.hiddenLayers.Count - 1, parent1.hiddenLayers[0].ColumnCount);
        child2.Initialize(parent2.inputLayer.ColumnCount, parent2.outputLayer.ColumnCount, parent2.hiddenLayers.Count - 1, parent2.hiddenLayers[0].ColumnCount);

        for (int i = 0; i < child1.weights.Count; i++)
        {
            for (int x = 0; x < child1.weights[i].RowCount; x++)
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

        for (int i = 0; i < child1.recurrentWeights.Count; i++)
        {
            for (int x = 0; x < child1.recurrentWeights[i].RowCount; x++)
            {
                for (int y = 0; y < child1.recurrentWeights[i].ColumnCount; y++)
                {
                    child1.recurrentWeights[i][x, y] = parent1.recurrentWeights[i][x, y];
                    child2.recurrentWeights[i][x, y] = parent2.recurrentWeights[i][x, y];

                    if (Random.Range(0.0f, 1.0f) < 0.5f)
                    {
                        (child2.recurrentWeights[i], child1.recurrentWeights) = (child1.recurrentWeights[i], child2.recurrentWeights);
                    }
                }
            }
        }

        for (int i = 0; i < child1.recurrentBiases.Count; i++)
        {
            child1.recurrentBiases[i] = parent1.recurrentBiases[i];
            child2.recurrentBiases[i] = parent2.recurrentBiases[i];

            if (Random.Range(0.0f, 1.0f) < 0.5f)
            {
                (child2.recurrentBiases[i], child1.recurrentBiases) = (child1.recurrentBiases[i], child2.recurrentBiases);
            }
        }

        return (child1, child2);
    }

    public new void Mutate(float mutationRate)
    {
        base.Mutate(mutationRate);

        for (int i = 0; i < recurrentWeights.Count; i++)
        {
            if (Random.Range(0.0f, 1.0f) < mutationRate)
            {
                int randomPoints = Random.Range(1, (recurrentWeights[i].RowCount * recurrentWeights[i].ColumnCount) / 7);

                for (int j = 0; j < randomPoints; j++)
                {
                    int randomColumn = Random.Range(0, recurrentWeights[i].ColumnCount);
                    int randomRow = Random.Range(0, recurrentWeights[i].RowCount);

                    recurrentWeights[i][randomRow, randomColumn] = Mathf.Clamp(recurrentWeights[i][randomRow, randomColumn] + Random.Range(-1f, 1f), -1f, 1f);
                }
            }
        }

        for (int i = 0; i < recurrentBiases.Count; i++)
        {
            if (Random.Range(0.0f, 1.0f) < mutationRate)
            {
                recurrentBiases[i] = Mathf.Clamp(recurrentBiases[i] + Random.Range(-1f, 1f), -1f, 1f);
            }
        }
    }

}
