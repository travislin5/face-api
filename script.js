const faceapi = require('@vladmandic/face-api');
const canvas = require('canvas');
const path = require('path');

// 設置 canvas
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

async function loadModels() {
  const modelPath = path.join(__dirname, 'model');
  console.log(modelPath)
  await faceapi.nets.tinyFaceDetector.loadFromDisk(modelPath);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);
  await faceapi.nets.faceExpressionNet.loadFromDisk(modelPath);
  await faceapi.nets.ageGenderNet.loadFromDisk(modelPath);
}

async function processImage(imagePath) {
  const img = await canvas.loadImage(imagePath);
  const detection = await faceapi
    .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions())
    .withFaceLandmarks()
    .withFaceDescriptor()
    .withFaceExpressions()
    .withAgeAndGender();
  
  if (!detection) {
    throw new Error(`No face detected in image: ${imagePath}`);
  }
  
  return detection;
}

(async () => {
  await loadModels();

  const image1Path = path.join(__dirname, 'img1.jpeg'); // img1 path
  const image2Path = path.join(__dirname, 'img2.jpeg'); // img2 path

  try {
    const [fd1, fd2] = await Promise.all([
      processImage(image1Path),
      processImage(image2Path),
    ]);

    const distance = faceapi.euclideanDistance(fd1.descriptor, fd2.descriptor);
    if (distance < 0.6) {
      console.log('Same person');
    } else {
      console.log('Different person');
    }

    console.log(`Distance: ${distance}`);
  } catch (error) {
    console.error(error);
  }
})();


// npm install @tensorflow/tfjs-node
// npm install @vladmandic/face-api
// npm install canvas