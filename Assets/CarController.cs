using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CarController : MonoBehaviour
{
    
    public LayerMask groundLayer;
    private Vector3 startPosition, startRotation;
    [Range(-1f, 1f)]
    public float accelaration, rotation;

    public float timeSinceStart = 0f;

    [Header("Fitness")]
    public float overallFitness;

    public float distanceWeight = 1.4f;
    public float avgSpeedWeight = 0.2f;

    public float avgSensorWeight = 0.1f;

    [Header("Network Options")]
    public int LAYERS = 1;
    public int NEURONS = 10;
    public int numberOfSensors = 5;

    List<float> sensors = new();
    private Vector3 lastPosition;
    private float totalDistanceTravelled;

    private float avgSpeed;
    private NeuralNetwork network;

    private void Awake()
    {
        startPosition = transform.position;
        startRotation = transform.eulerAngles;
        network = GetComponent<NeuralNetwork>();
    }

    public void Reset()
    {
        timeSinceStart = 0f;
        totalDistanceTravelled = 0f;
        avgSpeed = 0f;
        lastPosition = startPosition;
        overallFitness = 0f;
        transform.position = startPosition;
        transform.eulerAngles = startRotation;
    }

    public void ResetWithNetwork(NeuralNetwork net)
    {
        network = net;
        Reset();
    }
    private void OnCollisionEnter(Collision collision)
    {
        Death();
    }

    private Vector3 pos;
    public void MoveCar(float acc, float rotation)
    {
        pos = Vector3.Lerp(Vector3.zero, new Vector3(0, 0, acc * 11f), 0.02f);
        pos = transform.TransformDirection(pos);
        transform.position += pos;
        transform.eulerAngles += new Vector3(0, (rotation * 90) * 0.02f, 0);
    }

    private void InputSensors()
    {
        if (sensors.Count == 0)
        {
            for (int i = 0; i < numberOfSensors; i++)
            {
                sensors.Add(i + 1);
            }
        }
        for (int i = 0; i < numberOfSensors; i++)
        {
            float angle = Mathf.PI / 6.0f + ((float)i / (float)(numberOfSensors - 1)) * 2 * Mathf.PI / 3.0f;

            Vector3 direction = new Vector3(Mathf.Cos(angle), 0.0f, Mathf.Sin(angle));
            Ray ray = new(transform.position, transform.TransformDirection(direction));

            RaycastHit hit;
            System.Random random = new System.Random();
            float noise = (float)(random.NextDouble());
            if (Physics.Raycast(ray, out hit))
            {
                sensors[i] = (hit.distance) / 30.0f;
            }
            else
            {
                sensors[i] = 0;
            }
            sensors[i] += 0.3f * noise;
        }
    }

    private void FixedUpdate()
    {
        InputSensors();
        lastPosition = transform.position;
        (accelaration, rotation) = network.RunNetwork(sensors);
        MoveCar(accelaration, rotation);
        timeSinceStart += Time.deltaTime;
        CalculateFitness();
    }

    private bool isAboveTrack()
    {
        Ray ray = new Ray(transform.position, Vector3.down);
        float maxRayDistance = 100.0f;

        if (Physics.Raycast(ray, out RaycastHit hit, maxRayDistance))
        {
            GameObject objectBelowCar = hit.collider.gameObject;

            if (objectBelowCar.name == "Road")
                return true;
        }

        return false;
    }

    private void CalculateFitness()
    {
        totalDistanceTravelled += Vector3.Distance(transform.position, lastPosition);
        avgSpeed = totalDistanceTravelled / timeSinceStart;
        float avgSensor = 0;
        overallFitness = totalDistanceTravelled * distanceWeight + avgSpeed * avgSpeedWeight + avgSensor * avgSensorWeight;

        if (timeSinceStart > 20 && overallFitness < 50)
        {
            Death();
        }
        isAboveTrack();
        if (overallFitness > 1000)
        {
            Death();
        }
    }

    private void Death()
    {
        GameObject.FindObjectOfType<GeneticAlgorithmManager>().Death(overallFitness);
    }

}
