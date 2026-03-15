import { KEY_MAP } from './constants.js';

/**
 * Translates keyboard (and optional touch) input into game actions.
 * Decoupled from the game loop so it can be swapped or extended.
 */
export class InputHandler {
  constructor() {
    this._directionCallback = null;
    this._actionCallback = null;
    this._pauseCallback = null;
    this._bound = this._onKeyDown.bind(this);
  }

  /** Register callbacks and start listening. */
  bind({ onDirection, onAction, onPause }) {
    this._directionCallback = onDirection;
    this._actionCallback = onAction;
    this._pauseCallback = onPause;
    window.addEventListener('keydown', this._bound);
  }

  unbind() {
    window.removeEventListener('keydown', this._bound);
  }

  _onKeyDown(e) {
    const dir = KEY_MAP[e.key];
    if (dir) {
      e.preventDefault();
      this._directionCallback?.(dir);
      return;
    }

    if (e.key === ' ' || e.key === 'Enter') {
      e.preventDefault();
      this._actionCallback?.();
      return;
    }

    if (e.key === 'Escape' || e.key === 'p' || e.key === 'P') {
      e.preventDefault();
      this._pauseCallback?.();
    }
  }

  /**
   * Attach swipe listeners to an element for mobile support.
   * @param {HTMLElement} el
   */
  bindTouch(el) {
    let startX = 0;
    let startY = 0;
    const THRESHOLD = 30;

    el.addEventListener('touchstart', (e) => {
      const t = e.touches[0];
      startX = t.clientX;
      startY = t.clientY;
    }, { passive: true });

    el.addEventListener('touchend', (e) => {
      const t = e.changedTouches[0];
      const dx = t.clientX - startX;
      const dy = t.clientY - startY;

      if (Math.abs(dx) < THRESHOLD && Math.abs(dy) < THRESHOLD) {
        this._actionCallback?.();
        return;
      }

      if (Math.abs(dx) > Math.abs(dy)) {
        this._directionCallback?.(dx > 0 ? { x: 1, y: 0 } : { x: -1, y: 0 });
      } else {
        this._directionCallback?.(dy > 0 ? { x: 0, y: 1 } : { x: 0, y: -1 });
      }
    }, { passive: true });
  }
}
