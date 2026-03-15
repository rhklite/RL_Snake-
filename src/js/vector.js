/**
 * Lightweight immutable 2-D vector used for grid positions and directions.
 */
export class Vec2 {
  constructor(x, y) {
    this.x = x;
    this.y = y;
  }

  add(other) {
    return new Vec2(this.x + other.x, this.y + other.y);
  }

  equals(other) {
    return this.x === other.x && this.y === other.y;
  }

  wrap(cols, rows) {
    return new Vec2(
      ((this.x % cols) + cols) % cols,
      ((this.y % rows) + rows) % rows,
    );
  }

  static from(obj) {
    return new Vec2(obj.x, obj.y);
  }

  toString() {
    return `${this.x},${this.y}`;
  }
}
