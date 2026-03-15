import { GRID, CANVAS, COLORS } from './constants.js';

/**
 * Draws every visual element onto an HTML Canvas.
 * Stateless — give it data each frame and it renders.
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
    this.ctx.strokeStyle = COLORS.GRID_LINE;
    this.ctx.lineWidth = 1;
    for (let x = 0; x <= GRID.COLS; x++) {
      const px = x * this._cellSize;
      this.ctx.beginPath();
      this.ctx.moveTo(px, 0);
      this.ctx.lineTo(px, CANVAS.HEIGHT);
      this.ctx.stroke();
    }
    for (let y = 0; y <= GRID.ROWS; y++) {
      const py = y * this._cellSize;
      this.ctx.beginPath();
      this.ctx.moveTo(0, py);
      this.ctx.lineTo(CANVAS.WIDTH, py);
      this.ctx.stroke();
    }
  }

  /** @param {import('./snake.js').Snake} snake */
  drawSnake(snake) {
    const s = this._cellSize;
    const pad = 2;

    snake.segments.forEach((seg, i) => {
      const x = seg.x * s + pad;
      const y = seg.y * s + pad;
      const size = s - pad * 2;

      this.ctx.fillStyle = i === 0 ? COLORS.SNAKE_HEAD : COLORS.SNAKE_BODY;
      this.ctx.strokeStyle = COLORS.SNAKE_STROKE;
      this.ctx.lineWidth = 1.5;

      const radius = i === 0 ? size / 3 : size / 5;
      this._roundRect(x, y, size, size, radius);
      this.ctx.fill();
      this.ctx.stroke();
    });
  }

  /** @param {import('./food.js').Food} food */
  drawFood(food) {
    const s = this._cellSize;
    const cx = food.position.x * s + s / 2;
    const cy = food.position.y * s + s / 2;
    const r = s / 2 - 3;

    this.ctx.shadowColor = COLORS.FOOD_GLOW;
    this.ctx.shadowBlur = 12;

    this.ctx.fillStyle = COLORS.FOOD;
    this.ctx.beginPath();
    this.ctx.arc(cx, cy, r, 0, Math.PI * 2);
    this.ctx.fill();

    this.ctx.shadowBlur = 0;
  }

  drawScore(score, highScore) {
    this.ctx.fillStyle = COLORS.TEXT;
    this.ctx.font = 'bold 14px "Inter", system-ui, sans-serif';
    this.ctx.textAlign = 'left';
    this.ctx.fillText(`Score: ${score}`, 10, 20);
    this.ctx.textAlign = 'right';
    this.ctx.fillText(`Best: ${highScore}`, CANVAS.WIDTH - 10, 20);
  }

  drawOverlay(title, subtitle) {
    this.ctx.fillStyle = COLORS.OVERLAY;
    this.ctx.fillRect(0, 0, CANVAS.WIDTH, CANVAS.HEIGHT);

    this.ctx.fillStyle = COLORS.TEXT;
    this.ctx.textAlign = 'center';

    this.ctx.font = 'bold 28px "Inter", system-ui, sans-serif';
    this.ctx.fillText(title, CANVAS.WIDTH / 2, CANVAS.HEIGHT / 2 - 10);

    this.ctx.font = '14px "Inter", system-ui, sans-serif';
    this.ctx.fillText(subtitle, CANVAS.WIDTH / 2, CANVAS.HEIGHT / 2 + 20);
  }

  _roundRect(x, y, w, h, r) {
    this.ctx.beginPath();
    this.ctx.moveTo(x + r, y);
    this.ctx.lineTo(x + w - r, y);
    this.ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    this.ctx.lineTo(x + w, y + h - r);
    this.ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    this.ctx.lineTo(x + r, y + h);
    this.ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    this.ctx.lineTo(x, y + r);
    this.ctx.quadraticCurveTo(x, y, x + r, y);
    this.ctx.closePath();
  }
}
