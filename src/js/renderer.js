import { GRID, CANVAS, COLORS } from './constants.js';

/**
 * Draws every visual element onto an HTML Canvas.
 * Stateless per-frame — call methods in order each animation frame.
 */
export class Renderer {
  /** @param {HTMLCanvasElement} canvas */
  constructor(canvas) {
    canvas.width = CANVAS.WIDTH;
    canvas.height = CANVAS.HEIGHT;
    this.ctx = canvas.getContext('2d');
    this._cellSize = GRID.CELL_SIZE;
  }

  clear() {
    this.ctx.fillStyle = COLORS.BG;
    this.ctx.fillRect(0, 0, CANVAS.WIDTH, CANVAS.HEIGHT);
    this._drawGridLines();
  }

  _drawGridLines() {
    const { ctx } = this;
    ctx.strokeStyle = COLORS.GRID_LINE;
    ctx.lineWidth = 1;
    for (let x = 0; x <= GRID.COLS; x++) {
      const px = x * this._cellSize;
      ctx.beginPath();
      ctx.moveTo(px, 0);
      ctx.lineTo(px, CANVAS.HEIGHT);
      ctx.stroke();
    }
    for (let y = 0; y <= GRID.ROWS; y++) {
      const py = y * this._cellSize;
      ctx.beginPath();
      ctx.moveTo(0, py);
      ctx.lineTo(CANVAS.WIDTH, py);
      ctx.stroke();
    }
  }

  /** @param {import('./snake.js').Snake} snake */
  drawSnake(snake) {
    const { ctx } = this;
    const s = this._cellSize;
    const pad = 2;
    const len = snake.segments.length;

    for (let i = len - 1; i >= 0; i--) {
      const seg = snake.segments[i];
      const x = seg.x * s + pad;
      const y = seg.y * s + pad;
      const size = s - pad * 2;

      if (i === 0) {
        ctx.fillStyle = COLORS.SNAKE_HEAD;
      } else {
        const t = i / Math.max(len - 1, 1);
        ctx.fillStyle = lerpColor(COLORS.SNAKE_HEAD, COLORS.SNAKE_BODY, t);
      }

      ctx.strokeStyle = COLORS.SNAKE_STROKE;
      ctx.lineWidth = 1.5;

      const radius = i === 0 ? size / 3 : size / 5;
      this._roundRect(x, y, size, size, radius);
      ctx.fill();
      ctx.stroke();
    }
  }

  /**
   * @param {import('./food.js').Food} food
   * @param {number} timestamp - used for the pulsing glow animation
   */
  drawFood(food, timestamp = 0) {
    const { ctx } = this;
    const s = this._cellSize;
    const cx = food.position.x * s + s / 2;
    const cy = food.position.y * s + s / 2;
    const baseR = s / 2 - 3;

    const pulse = 1 + 0.12 * Math.sin(timestamp / 300);

    ctx.shadowColor = COLORS.FOOD_GLOW;
    ctx.shadowBlur = 10 + 6 * pulse;

    ctx.fillStyle = COLORS.FOOD;
    ctx.beginPath();
    ctx.arc(cx, cy, baseR * pulse, 0, Math.PI * 2);
    ctx.fill();

    ctx.shadowBlur = 0;
  }

  drawScore(score, highScore) {
    const { ctx } = this;
    ctx.fillStyle = COLORS.TEXT;
    ctx.font = 'bold 14px "Inter", system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`Score: ${score}`, 10, 20);
    ctx.textAlign = 'right';
    ctx.fillText(`Best: ${highScore}`, CANVAS.WIDTH - 10, 20);
  }

  drawOverlay(title, subtitle) {
    const { ctx } = this;
    ctx.fillStyle = COLORS.OVERLAY;
    ctx.fillRect(0, 0, CANVAS.WIDTH, CANVAS.HEIGHT);

    ctx.fillStyle = COLORS.TEXT;
    ctx.textAlign = 'center';

    ctx.font = 'bold 28px "Inter", system-ui, sans-serif';
    ctx.fillText(title, CANVAS.WIDTH / 2, CANVAS.HEIGHT / 2 - 10);

    ctx.font = '14px "Inter", system-ui, sans-serif';
    ctx.fillText(subtitle, CANVAS.WIDTH / 2, CANVAS.HEIGHT / 2 + 20);
  }

  _roundRect(x, y, w, h, r) {
    const { ctx } = this;
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }
}

/* ---- colour helpers ---- */

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff];
}

function lerpColor(hexA, hexB, t) {
  const a = hexToRgb(hexA);
  const b = hexToRgb(hexB);
  const r = Math.round(a[0] + (b[0] - a[0]) * t);
  const g = Math.round(a[1] + (b[1] - a[1]) * t);
  const bl = Math.round(a[2] + (b[2] - a[2]) * t);
  return `rgb(${r},${g},${bl})`;
}
