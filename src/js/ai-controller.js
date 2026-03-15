import { StateEncoder } from './state-encoder.js';

/**
 * Loads an ONNX model via ONNX Runtime Web and drives the snake each tick.
 *
 * Usage:
 *   const ai = new AIController();
 *   await ai.loadModel(onnxFileOrUrl);
 *   // each game tick:
 *   ai.step(game);          // encodes state → inference → setDirection
 *
 * The controller expects the global `ort` namespace provided by the
 * ONNX Runtime Web CDN script (loaded in index.html).
 *
 * Model contract:
 *   Input  — float32 tensor [1, 11]  (see StateEncoder)
 *   Output — float32 tensor [1, 3]   (logits for [straight, right, left])
 */
export class AIController {
  constructor(encoder = new StateEncoder()) {
    this.encoder = encoder;
    this._session = null;
    this._inputName = null;
    this.ready = false;
  }

  /**
   * Load an ONNX model from a File object, ArrayBuffer, or URL string.
   * @param {File | ArrayBuffer | string} source
   */
  async loadModel(source) {
    if (typeof ort === 'undefined') {
      throw new Error(
        'ONNX Runtime Web is not loaded. ' +
        'Add the ort CDN script to index.html before using AIController.',
      );
    }

    let buffer;
    if (source instanceof File) {
      buffer = await source.arrayBuffer();
    } else if (source instanceof ArrayBuffer) {
      buffer = source;
    } else if (typeof source === 'string') {
      const resp = await fetch(source);
      buffer = await resp.arrayBuffer();
    } else {
      throw new TypeError('source must be a File, ArrayBuffer, or URL string');
    }

    this._session = await ort.InferenceSession.create(buffer, {
      executionProviders: ['wasm'],
    });

    this._inputName = this._session.inputNames[0];
    this.ready = true;
  }

  /**
   * Observe the current game state, run inference, and apply the chosen
   * direction to the snake. Call this once per game tick before `game.tick()`.
   *
   * @param {import('./game.js').Game} game
   */
  async step(game) {
    if (!this.ready) return;

    const features = this.encoder.encode(game.snake, game.food);

    const inputTensor = new ort.Tensor('float32', features, [1, this.encoder.featureSize]);
    const results = await this._session.run({ [this._inputName]: inputTensor });

    const outputName = this._session.outputNames[0];
    const logits = results[outputName].data;

    let bestIdx = 0;
    for (let i = 1; i < logits.length; i++) {
      if (logits[i] > logits[bestIdx]) bestIdx = i;
    }

    const dir = this.encoder.actionToDirection(bestIdx, game.snake.direction);
    game.snake.setDirection(dir);
  }

  dispose() {
    this._session?.release();
    this._session = null;
    this.ready = false;
  }
}
