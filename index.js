import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis'

import {ObjectDetectionImageSynthesizer} from './synthetic_images';
import { loadGraphModel } from '@tensorflow/tfjs';

const canvas = document.getElementById('data-canvas');
const status = document.getElementById('status');
const testModel = document.getElementById('test');
const loadHostedModel = document.getElementById('load-hosted-model');
const inferenceTimeMs = document.getElementById('inference-time-ms');
const trueObjectClass = document.getElementById('true-object-class');
const predictedObjectClass = document.getElementById('predicted-object-class');

const loadModelDiv = document.getElementById('loadmodels');

const TRUE_BOUNDING_BOX_LINE_WIDTH = 2;
const TRUE_BOUNDING_BOX_STYLE = 'rgb(255,0,0)';
const PREDICT_BOUNDING_BOX_LINE_WIDTH = 2;
const PREDICT_BOUNDING_BOX_STYLE = 'rgb(0,0,255)';

function drawBoundingBoxes(canvas, trueBoundingBox, predictBoundingBox) {
  tf.util.assert(
      trueBoundingBox != null && trueBoundingBox.length === 4,
      `Expected boundingBoxArray to have length 4, ` +
          `but got ${trueBoundingBox} instead`);
  tf.util.assert(
      predictBoundingBox != null && predictBoundingBox.length === 4,
      `Expected boundingBoxArray to have length 4, ` +
          `but got ${trueBoundingBox} instead`);

  let left = trueBoundingBox[0];
  let right = trueBoundingBox[1];
  let top = trueBoundingBox[2];
  let bottom = trueBoundingBox[3];

  const ctx = canvas.getContext('2d');
  ctx.beginPath();
  ctx.strokeStyle = TRUE_BOUNDING_BOX_STYLE;
  ctx.lineWidth = TRUE_BOUNDING_BOX_LINE_WIDTH;
  ctx.moveTo(left, top);
  ctx.lineTo(right, top);
  ctx.lineTo(right, bottom);
  ctx.lineTo(left, bottom);
  ctx.lineTo(left, top);
  ctx.stroke();

  ctx.font = '15px Arial';
  ctx.fillStyle = TRUE_BOUNDING_BOX_STYLE;
  ctx.fillText('true', left, top);

  left = predictBoundingBox[0];
  right = predictBoundingBox[1];
  top = predictBoundingBox[2];
  bottom = predictBoundingBox[3];

  ctx.beginPath();
  ctx.strokeStyle = PREDICT_BOUNDING_BOX_STYLE;
  ctx.lineWidth = PREDICT_BOUNDING_BOX_LINE_WIDTH;
  ctx.moveTo(left, top);
  ctx.lineTo(right, top);
  ctx.lineTo(right, bottom);
  ctx.lineTo(left, bottom);
  ctx.lineTo(left, top);
  ctx.stroke();

  ctx.font = '15px Arial';
  ctx.fillStyle = PREDICT_BOUNDING_BOX_STYLE;
  ctx.fillText('predicted', left, bottom);
}

/**
 * Synthesize an input image, run inference on it and visualize the results.
 *
 * @param {tf.Model} model Model to be used for inference.
 */
async function runAndVisualizeInference(model) {
  // Synthesize an input image and show it in the canvas.
  const synth = new ObjectDetectionImageSynthesizer(canvas, tf);

  const numExamples = 1;
  const numCircles = 10;
  const numLineSegments = 10;
  const {images, targets} = await synth.generateExampleBatch(
      numExamples, numCircles, numLineSegments);

  const t0 = tf.util.now();
  // Runs inference with the model.
  const modelOut = await model.predict(images).data();
  inferenceTimeMs.textContent = `${(tf.util.now() - t0).toFixed(1)}`;

  // Visualize the true and predicted bounding boxes.
  const targetsArray = Array.from(await targets.data());
  const boundingBoxArray = targetsArray.slice(1);
  drawBoundingBoxes(canvas, boundingBoxArray, modelOut.slice(1));

  // Display the true and predict object classes.
  const trueClassName = targetsArray[0] > 0 ? 'rectangle' : 'triangle';
  trueObjectClass.textContent = trueClassName;

  // The model predicts a number to indicate the predicted class
  // of the object. It is trained to predict 0 for triangle and
  // 224 (canvas.width) for rectangel. This is how the model combines
  // the class loss with the bounding-box loss to form a single loss
  // value. Therefore, at inference time, we threshold the number
  // by half of 224 (canvas.width).
  const shapeClassificationThreshold = canvas.width / 2;
  const predictClassName =
      (modelOut[0] > shapeClassificationThreshold) ? 'rectangle' : 'triangle';
  predictedObjectClass.textContent = predictClassName;

  if (predictClassName === trueClassName) {
    predictedObjectClass.classList.remove('shape-class-wrong');
    predictedObjectClass.classList.add('shape-class-correct');
  } else {
    predictedObjectClass.classList.remove('shape-class-correct');
    predictedObjectClass.classList.add('shape-class-wrong');
  }

  // Tensor memory cleanup.
  tf.dispose([images, targets]);
}

async function init() {

   const visor = tfvis.visor()

  const HOSTED_MODEL_PATH = './model/mobilenet/model.json' //'https://storage.googleapis.com/tfjs-examples/simple-object-detection/dist/object_detection_model/model.json';
  const TRAINING_MODEL_PATH = './object_detection_model/model.json';
  const RUN_NO_TRAIN_MODEL_PATH = './model/run-no-train/model.json';
  const RUN1_MODEL_PATH = './model/run1/model.json';
  const RUN2_MODEL_PATH = './model/run2/model.json';
  
  //const  MOBILENETV1 = 'https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v1_1.0_224/model.json'

  let models = {
    mobilenet: 
      {
        description: "Base Mobile Net Performance",
        path: HOSTED_MODEL_PATH
      } ,
    latest:   {
      description: "Latest Training",
      path: TRAINING_MODEL_PATH
     },
    no_train:  {
      description: "Model with no training",
      path: RUN_NO_TRAIN_MODEL_PATH
     }, 
     run1: {
       description: "Model with significant training(training images=2000,batchsize=32epochs=200,finetune=100)",
       path: './model/run2/model.json'
     },
     run2: {
        description: "Model with significant training(training images=1000,batchsize=16 epochs=100,finetune=100)",
        path: './model/run2/model.json'
      }
  }

  let keys = Object.keys(models)

  // Attempt to load locally-saved model. If it fails, activate the
  // "Load hosted model" button.
  let model, trainingModel,mobileNetModel;

  window.loadModel= async (index, el)=>{ 
    let modelMeta = models[index]
    let modelPath = modelMeta.path

    keys.forEach((modelKey)=>{
      let e = document.getElementById(modelKey)
      e.disabled=false
   })

    el.disabled = true;

    try {
      status.textContent = `Loading model from ${modelPath} ...`;
      model = await tf.loadLayersModel(modelPath);
  
      const surface = { name: 'Model Summary', tab: 'Trained Model'};
      tfvis.show.modelSummary(surface,model)
  
      //model.summary();
      testModel.disabled = false;
      status.textContent = 'Loaded model! Now click "Test Model".';
      runAndVisualizeInference(model);
    } catch (err) {
      status.textContent = 'Failed to load model. ' +
          'Please click "Load Hosted Model"';
    }
  }


  let h = keys.map((modelKey)=>{
    let modelMeta = models[modelKey]
    return `<button id='${modelKey}' onClick="loadModel('${modelKey}',this)">${modelMeta.description}</button>`
  })
  loadModelDiv.innerHTML = h
                        
  testModel.addEventListener('click', () => runAndVisualizeInference(model));
}

init();
