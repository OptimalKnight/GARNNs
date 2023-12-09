using UnityEngine;
public class MoveCube : MonoBehaviour
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

        transform.position += new Vector3((float)0.5 * Time.deltaTime * factor, 0, (float)0.25 * Time.deltaTime * factor);
    }
    
}
