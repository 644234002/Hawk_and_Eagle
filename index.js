
  async function imageShow() {
    const inputElement = document.getElementById('imageFileInput');
    const image = inputElement.files[0];
    const imgElement = document.getElementById('imageInput');
    imgElement.src = URL.createObjectURL(image);
    imgElement.srcObject = image;
    predictImage();
  }



  async function predictImage() {

    const labels = ['Hawk', 'Eagle']; // Replace with actual class labels

    const model = await tflite.loadTFLiteModel(
      "./models/Hawk_and_Eagle.tflite",
    );

    
    // Load and preprocess the uploaded image
    const imgElement = document.querySelector("img");
    

    const tensor = tf.browser.fromPixels(imgElement).toFloat();
    const resizedTensor = tf.image.resizeBilinear(tensor, [128, 128]) 
      .expandDims()
      .div(127.5)
      .sub(1);
    const preprocessedTensor = resizedTensor.expandDims(0);

    
    // Run inference on the preprocessed image
    const predictions = await model.predict(resizedTensor);


    // Process the predictions and display the results (Multi Label Classification)
    //const topPredictionIndex = predictions.argMax(1).dataSync()[0];
    //const topPredictionScore = predictions.dataSync()[topPredictionIndex];
    //const predictedLabel = labels[topPredictionIndex];

    // Process the predictions and display the results (Binary Classification)
    const topPredictionScore = predictions.dataSync()[0];
    const predictedLabel = topPredictionScore < 0.5 ? labels[0] : labels[1];


    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = `
      <p>Predicted Label: <b>${predictedLabel}</b></p>
      <p>Predicted Prob: <b>${topPredictionScore.toFixed(4)}</b></p>
      <p><b>Note:</b> Prob < <b>0.5</b>, predicted as Eagle.<br/> Prob >= <b>0.5</b>, predicted as Hawk. </p>
    `;

    // Dispose of tensors to free up memory
    tensor.dispose();
    resizedTensor.dispose();
    preprocessedTensor.dispose();

   
  }
  
