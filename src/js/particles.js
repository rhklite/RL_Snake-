import { GRID } from './constants.js';

/**
 * Lightweight particle system for eat / death visual feedback.
 * Particles are fire-and-forget; they expire after their lifetime.
 */
export class ParticleSystem {
  constructor() {
    this._particles = [];
  }

  /** Spawn a burst of particles at a grid cell. */
  burst(gridX, gridY, color, count = 8) {
    const s = GRID.CELL_SIZE;
    const cx = gridX * s + s / 2;
    const cy = gridY * s + s / 2;

    for (let i = 0; i < count; i++) {
      const angle = (Math.PI * 2 * i) / count + Math.random() * 0.4;
      const speed = 40 + Math.random() * 60;
      this._particles.push({
        x: cx,
        y: cy,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        life: 1,
        decay: 1.5 + Math.random() * 1.5,
        radius: 2 + Math.random() * 2,
        color,
      });
    }
  }

  /** Advance and cull dead particles. */
  update(dt) {
    for (let i = this._particles.length - 1; i >= 0; i--) {
      const p = this._particles[i];
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.life -= p.decay * dt;
      if (p.life <= 0) {
        this._particles.splice(i, 1);
      }
    }
  }

  /** @param {CanvasRenderingContext2D} ctx */
  draw(ctx) {
    for (const p of this._particles) {
      ctx.globalAlpha = Math.max(0, p.life);
      ctx.fillStyle = p.color;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius * p.life, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  get active() {
    return this._particles.length > 0;
  }
}
