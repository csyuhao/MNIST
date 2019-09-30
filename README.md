# MNIST Model

## Project File Structure
- `root`
   - `data`         // mnist dataset storage
   - `img`          // handwritten digit image samples
     - 0 - 7.jpg    // index - digit pair
   - `model`        // pretrained model
   - `perturb`      // FGSM generated adversarial examples
   - `attack.py`    // FGSM algorithm
   - `get_mnist.py` // get handwritten digit image samples
   -` predict.py`   // load pretrained model to predict images
   - `train.py`     // train model using datasets of `data`