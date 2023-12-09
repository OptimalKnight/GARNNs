using UnityEngine;

public class MoveCube2 : MonoBehaviour
{

    int counter, factor;

    private void Start()
    {
        counter = 0;
        factor = -1;
    }

    private void Update()
    {
        counter += 1;
        if (counter % 1000 == 0)
        {
            factor *= -1;
        }

        transform.position += new Vector3((float)-0.75 * Time.deltaTime * factor, 0, (float)0.25 * Time.deltaTime * factor);
    }
    
}
