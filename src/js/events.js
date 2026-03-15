/**
 * Minimal pub/sub event emitter.
 * Keeps Game decoupled from any specific UI layer.
 */
export class EventEmitter {
  constructor() {
    this._listeners = new Map();
  }

  on(event, fn) {
    if (!this._listeners.has(event)) {
      this._listeners.set(event, []);
    }
    this._listeners.get(event).push(fn);
    return () => this.off(event, fn);
  }

  off(event, fn) {
    const list = this._listeners.get(event);
    if (!list) return;
    const idx = list.indexOf(fn);
    if (idx !== -1) list.splice(idx, 1);
  }

  emit(event, ...args) {
    const list = this._listeners.get(event);
    if (list) list.forEach((fn) => fn(...args));
  }
}
