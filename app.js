const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const faceapi = require('@vladmandic/face-api');
const canvas = require('canvas');

const app = express();
const upload = multer({ dest: 'uploads/' });

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

  app.use(express.static(path.join(__dirname, 'public')));

  app.post('/compare', upload.fields([{ name: 'image1' }, { name: 'image2' }]), async (req, res) => {
    try {
      const image1Path = req.files.image1[0].path;
      const image2Path = req.files.image2[0].path;

      const [fd1, fd2] = await Promise.all([
        processImage(image1Path),
        processImage(image2Path),
      ]);

      const distance = faceapi.euclideanDistance(fd1.descriptor, fd2.descriptor);
      const result = distance < 0.6 ? 'Same person' : 'Different person';

      // 刪除上傳的圖片
      fs.unlinkSync(image1Path);
      fs.unlinkSync(image2Path);

      res.json({ distance, result });
    } catch (error) {
      console.error(error);
      res.status(500).send({ error: error.message });
    }
  });

  const PORT = process.env.PORT || 3000;
  app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
  });
})();
