---
layout: post
title: "Training a multi-layer perceptron on an Arduino"
date: 2026-10-29
category: "Embedded, Arduino"
description: Building and training a multi-layer perceptron on an Arduino UNO and displaying the neurons with LEDs
---
The perceptron is the most fundamental unit of a deep learning model and stacking perceptron layers gives you a multi-layer perceptron. Building one for an embedded system like the Arduino UNO helped me develop a deeper understanding of how it worked, since every element of it had to be reduced to simple code. I first evaluated the model by reading its serial logs and then representing each neuron with its own LED on a breadboard.
## The Architecture 

<div class="imgbox">
<img alt="mlp.png" src="/assets/images/mlp.png"><br>
<span class="img_subtext">The MLP Architecure. Drawn with ChatGPT</span>
</div>
I started with an input layer of three inputs, followed by a fully connected hidden layer having four neurons and finally a single output neuron. The activation function I used is sigmoid.

## Initialization

Number of training cases = N \
Inputs = X (N x 3)\
Unactivated Hidden Layer = Z<sub>1</sub> (N x 4)\
Activated Hidden Layer = A<sub>1</sub> (N x 4)\
Unactivated Output Layer = Z<sub>2</sub> (N x 1)\
Activated Output Layer = A<sub>1</sub> (N x 1)\
Expected Outputs = ŷ (N x 1)


We denote the input to hidden layer weights and biases by W<sub>1</sub> and B<sub>1</sub> respectively.
Similarly, the hidden to output layer weights and biases are W<sub>2</sub> and B<sub>2</sub> respectively.

W<sub>1</sub> -> 3 x 4 \
B<sub>1</sub> -> 1 x 4 \
W<sub>2</sub> -> 4 x 1 \
B<sub>2</sub> -> 1 x 1

We also need to randomize them.

We keep a learning rate of 0.5 and 3000 epochs (executes quite slowly on an Arduino UNO).

Implementation in code:

{% raw %}
```cpp
const int numInputs = 3;
const int numHidden = 4;

const float lr = 0.5f;
const int epochs = 3000;

constexpr int N = 6;
const float z0[N][numInputs] = {{1,0,1}, {0,0,0}, {1,1,0}, {1,0,0}, {0,0,1}, {0,1,1}};
const float allInputs[8][numInputs] = {{1,0,1}, {0,0,0}, {1,1,0}, {1,0,0}, {0,0,1}, {0,1,1}, {1,1,1}, {0,1,0}};
const float ycap[N] = {1,0,1,1,0,0};

float w1[numInputs][numHidden];
float w2[numHidden];
float b1[numHidden];
float b2;

float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

float sigmoid_deriv(float a) {
  return a * (1.0f - a);
}

float randWeight() {
  return random(-1000, 1001) / 1000.0f;
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {}

  randomSeed(analogRead(A0));   // leave A0 unconnected
  // Assigning random weights and biases
  for (int i = 0; i < numInputs; i++)
    for (int j = 0; j < numHidden; j++)
      w1[i][j] = randWeight();
  for (int j = 0; j < numHidden; j++) {
    w2[j] = randWeight();
    b1[j] = randWeight();
  }
  b2 = randWeight();
  .
  .
  .
}
```
{% endraw %}

## Forward Propagation

Z<sub>1</sub> = X W<sub>1</sub> + B<sub>1</sub>\
A<sub>1</sub> = 𝜎(Z<sub>1</sub>)

Z<sub>2</sub> = A<sub>1</sub> W<sub>2</sub> + B<sub>2</sub>\
A<sub>2</sub> = 𝜎(Z<sub>2</sub>)

{% raw %}
```cpp
for (int x = 0; x < N; x++) {
    // Forward prop
    float a1[numHidden];
    for (int j = 0; j < numHidden; j++) {
        float z1 = b1[j];
        for (int k = 0; k < numInputs; k++)
            z1 += z0[x][k] * w1[k][j];
        a1[j] = sigmoid(z1);
    }
    float z2 = b2;
    for (int j = 0; j < numHidden; j++)
        z2 += a1[j] * w2[j];
    float a2 = sigmoid(z2);
    .
    .
    .
}
```
{% endraw %}

I use `float` instead of `double` to save memory.

## Backward Propagation

Now, we correct our weights and biases.

$$
\delta_2 = (a_2-y_x)\sigma'(z_2)
$$

$$
\delta_{1,j} = w_{2,j}\delta_2\sigma'(z_{1,j})
$$

$$
w_{2,j} \leftarrow w_{2,j}-\eta\delta_2a_{1,j}
$$

$$
b_2 \leftarrow b_2-\eta\delta_2
$$

$$
w_{1,kj} \leftarrow w_{1,kj}-\eta\delta_{1,j}x_k
$$

$$
b_{1,j} \leftarrow b_{1,j}-\eta\delta_{1,j}
$$

In code:

{% raw %}
```cpp
.
.
.
// Backprop
float delta2 = (a2 - ycap[x]) * sigmoid_deriv(a2);

float delta1[numHidden];
for (int j = 0; j < numHidden; j++)
    delta1[j] = w2[j] * delta2 * sigmoid_deriv(a1[j]);

for (int j = 0; j < numHidden; j++)
    w2[j] -= lr * delta2 * a1[j];
b2 -= lr * delta2;

for (int j = 0; j < numHidden; j++) {
    for (int k = 0; k < numInputs; k++)
        w1[k][j] -= lr * delta1[j] * z0[x][k];
    b1[j] -= lr * delta1[j];
}
.
.
.
}
```
{% endraw %}

## Visualising the MLP using LEDs

I arranged the 8 LEDs on a breadboard with 220 ohm resistors. Here's how I connected the LEDs to the Arduino UNO: 

<div class="imgbox">
<img alt="mlp.png" src="/assets/images/connections.jpeg"><br>
<span class="img_subtext">Sorry for the wire mess, the code below should make it easier to understand.</span>
</div>

Then, I added the following lines to the beginning

```cpp
const int inputLEDPins[numInputs] = {2,4,7};
const int hiddenLEDPins[numHidden] = {3,5,6,9};
const int outputLEDPin = 10;
int currentInput = 0; // This will make sense later
```

We try to visualize the changes in activation during backprop as it happens

```cpp
.
.
.
if (epoch % 100 == 0) {
    float* test = allInputs[currentInput];
    for (int j = 0; j < numInputs; j++) {
        int output = 255.0 * test[j];
        analogWrite(inputLEDPins[j], output);
    }
    float a1[numHidden];
    for (int j = 0; j < numHidden; j++) {
        float z1 = b1[j];
        for (int k = 0; k < numInputs; k++)
            z1 += test[k] * w1[k][j];
        a1[j] = sigmoid(z1);
        int output = 255.0 * a1[j];
        analogWrite(hiddenLEDPins[j], output);
    }
    float z2 = b2;
    for (int j = 0; j < numHidden; j++)
        z2 += a1[j] * w2[j];
    int output = 255.0 * sigmoid(z2);
    analogWrite(outputLEDPin, output);
}
if (epoch % 300 == 0) { // Serial logging
    Serial.print("Epoch: ");
    Serial.print(epoch);
    Serial.print("  Loss: ");
    Serial.println(loss, 6);
    currentInput += 1;
    currentInput = currentInput % 8;
}
```

Once trained, we flash all the LEDs four times to indicate completion.

```cpp
.
.
.
for (int i = 0; i < 8; i++) {
    int k = 255 * (i%2);
    for (int j = 0; j < numInputs; j++) {
        analogWrite(inputLEDPins[j], k);
    }
    for (int j = 0; j < numHidden; j++) {
        analogWrite(hiddenLEDPins[j], k);
    }
    analogWrite(outputLEDPin, k);
    delay(300);
}
```

Finally, we loop through each of the possible inputs indefinitely

```cpp
void loop() {
  for (int i = 0; i < 8; i++) {
    float* test = allInputs[i];
    for (int j = 0; j < numInputs; j++) {
      int output = 255.0 * test[j];
      analogWrite(inputLEDPins[j], output);
    }
    float a1[numHidden];
    for (int j = 0; j < numHidden; j++) {
      float z1 = b1[j];
      for (int k = 0; k < numInputs; k++)
        z1 += test[k] * w1[k][j];
      a1[j] = sigmoid(z1);
      int output = 255.0 * a1[j];
      analogWrite(hiddenLEDPins[j], output);
    }
    float z2 = b2;
    for (int j = 0; j < numHidden; j++)
      z2 += a1[j] * w2[j];
    int output = 255.0 * sigmoid(z2);
    analogWrite(outputLEDPin, output);
    Serial.print("Output: ");
    Serial.println(sigmoid(z2), 6);
    delay(1500);
  }
}
```

## The Complete Code

{% raw %}

```cpp
#include <math.h>

const int numInputs = 3;
const int numHidden = 4;

const float lr = 0.5f;
const int epochs = 3000;

constexpr int N = 6;
const float z0[N][numInputs] = {{1,0,1}, {0,0,0}, {1,1,0}, {1,0,0}, {0,0,1}, {0,1,1}};
const float allInputs[8][numInputs] = {{1,0,1}, {0,0,0}, {1,1,0}, {1,0,0}, {0,0,1}, {0,1,1}, {1,1,1}, {0,1,0}};
const float ycap[N] = {1,0,1,1,0,0};

int currentInput = 0;

float w1[numInputs][numHidden];
float w2[numHidden];
float b1[numHidden];
float b2;

const int inputLEDPins[numInputs] = {2,4,7};
const int hiddenLEDPins[numHidden] = {3,5,6,9};
const int outputLEDPin = 10;

float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

float sigmoid_deriv(float a) {
  return a * (1.0f - a);
}

// random float in [-1, 1]
float randWeight() {
  return random(-1000, 1001) / 1000.0f;
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {}            // needed on some boards (Leonardo, etc.)

  randomSeed(analogRead(A0));   // leave A0 unconnected, or use a fixed number

  for (int i = 0; i < numInputs; i++)
    for (int j = 0; j < numHidden; j++)
      w1[i][j] = randWeight();
  for (int j = 0; j < numHidden; j++) {
    w2[j] = randWeight();
    b1[j] = randWeight();
  }
  b2 = randWeight();

  for (int epoch = 0; epoch < epochs; epoch++) {
    float loss = 0;

    for (int x = 0; x < N; x++) {
      // Forward prop
      float a1[numHidden];
      for (int j = 0; j < numHidden; j++) {
        float z1 = b1[j];
        for (int k = 0; k < numInputs; k++)
          z1 += z0[x][k] * w1[k][j];
        a1[j] = sigmoid(z1);
      }
      float z2 = b2;
      for (int j = 0; j < numHidden; j++)
        z2 += a1[j] * w2[j];
      float a2 = sigmoid(z2);

      loss += 0.5f * (a2 - ycap[x]) * (a2 - ycap[x]);

      // Backprop
      float delta2 = (a2 - ycap[x]) * sigmoid_deriv(a2);

      float delta1[numHidden];
      for (int j = 0; j < numHidden; j++)
        delta1[j] = w2[j] * delta2 * sigmoid_deriv(a1[j]);

      for (int j = 0; j < numHidden; j++)
        w2[j] -= lr * delta2 * a1[j];
      b2 -= lr * delta2;

      for (int j = 0; j < numHidden; j++) {
        for (int k = 0; k < numInputs; k++)
          w1[k][j] -= lr * delta1[j] * z0[x][k];
        b1[j] -= lr * delta1[j];
      }
    }
    if (epoch % 100 == 0) {
      float* test = allInputs[currentInput];
      for (int j = 0; j < numInputs; j++) {
        int output = 255.0 * test[j];
        analogWrite(inputLEDPins[j], output);
      }
      float a1[numHidden];
      for (int j = 0; j < numHidden; j++) {
        float z1 = b1[j];
        for (int k = 0; k < numInputs; k++)
          z1 += test[k] * w1[k][j];
        a1[j] = sigmoid(z1);
        int output = 255.0 * a1[j];
        analogWrite(hiddenLEDPins[j], output);
      }
      float z2 = b2;
      for (int j = 0; j < numHidden; j++)
        z2 += a1[j] * w2[j];
      int output = 255.0 * sigmoid(z2);
      analogWrite(outputLEDPin, output);
    }
    if (epoch % 300 == 0) {
      Serial.print("Epoch: ");
      Serial.print(epoch);
      Serial.print("  Loss: ");
      Serial.println(loss, 6);
      currentInput += 1;
      currentInput = currentInput % 8;
    }
  }
  
  for (int i = 0; i < 8; i++) {
    int k = 255 * (i%2);
    for (int j = 0; j < numInputs; j++) {
      analogWrite(inputLEDPins[j], k);
    }
    for (int j = 0; j < numHidden; j++) {
      analogWrite(hiddenLEDPins[j], k);
    }
    analogWrite(outputLEDPin, k);
    delay(300);
  }
}

void loop() {
  for (int i = 0; i < 8; i++) {
    float* test = allInputs[i];
    for (int j = 0; j < numInputs; j++) {
      int output = 255.0 * test[j];
      analogWrite(inputLEDPins[j], output);
    }
    float a1[numHidden];
    for (int j = 0; j < numHidden; j++) {
      float z1 = b1[j];
      for (int k = 0; k < numInputs; k++)
        z1 += test[k] * w1[k][j];
      a1[j] = sigmoid(z1);
      int output = 255.0 * a1[j];
      analogWrite(hiddenLEDPins[j], output);
    }
    float z2 = b2;
    for (int j = 0; j < numHidden; j++)
      z2 += a1[j] * w2[j];
    int output = 255.0 * sigmoid(z2);
    analogWrite(outputLEDPin, output);
    Serial.print("Output: ");
    Serial.println(sigmoid(z2), 6);
    delay(1500);
  }
}
```
{% endraw %}

## Conclusion

This project gave me a deeper understanding of the backpropagation process as well as computing on embedded systems. I don't think I'll be exploring any more AI-ML tasks on the Arduino itself, but I will be using deep learning in my next project.