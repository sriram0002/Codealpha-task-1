#Hand writing detecting 
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>MNIST CNN Prediction with TensorFlow.js</title>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
  <style>
    body { font-family: Arial, sans-serif; text-align: center; }
    canvas { margin: 5px; border: 1px solid #ccc; }
    .digit-block { display: inline-block; margin: 10px; }
    .prediction { font-size: 18px; font-weight: bold; margin-top: 5px; }
  </style>
</head>
<body>
  <h1>MNIST Digit Prediction</h1>
  <div id="container"></div>

  <script>
    async function run() {
      const model = await tf.loadLayersModel('model/model.json'); // Replace with actual model URL
      const container = document.getElementById('container');

      for (let i = 0; i < 10; i++) {
        // Simulate a random 28x28 grayscale image (replace with real image in practice)
        const imgData = tf.randomUniform([28, 28], 0, 1);
        const input = imgData.reshape([1, 28, 28, 1]);
        const prediction = model.predict(input);
        const predDigit = prediction.argMax(-1).dataSync()[0];

        // Create canvas to display image
        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(28, 28);

        const data = imgData.dataSync();
        for (let j = 0; j < 28 * 28; j++) {
          const val = Math.floor(data[j] * 255);
          imageData.data[j * 4 + 0] = val; // R
          imageData.data[j * 4 + 1] = val; // G
          imageData.data[j * 4 + 2] = val; // B
          imageData.data[j * 4 + 3] = 255; // A
        }
        ctx.putImageData(imageData, 0, 0);

        // Create prediction label
        const predDiv = document.createElement('div');
        predDiv.className = 'prediction';
        predDiv.textContent = `Prediction: ${predDigit}`;

        // Group canvas + label
        const block = document.createElement('div');
        block.className = 'digit-block';
        block.appendChild(canvas);
        block.appendChild(predDiv);
        container.appendChild(block);
      }
    }

    run();
  </script>
</body>
</html>