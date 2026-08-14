#include "Physics2D.h"

#include <algorithm>
#include <cmath>

namespace shp {
namespace phys {
namespace {

// Baumgarte-style positional correction. Pushing the full overlap out in one
// step injects energy and makes piles jitter; 20% per step settles quietly.
constexpr float kBaumgarte = 0.2f;
// Overlap this small is left alone, so resting contacts stop twitching.
constexpr float kSlop = 0.05f;
constexpr int kVelocityIterations = 8;
// Below this closing speed a contact is treated as resting and gets no bounce.
// Applying restitution to a settled piece is what makes a pile hum forever.
constexpr float kRestingSpeed = 1.0f;

// A body's hull in world space.
void transformHull(const Body& b, std::vector<Vec2>& out) {
  out.resize(b.local.size());
  const float c = cosf(b.angle);
  const float s = sinf(b.angle);
  const Vec2 p = b.worldPos();
  for (std::size_t i = 0; i < b.local.size(); ++i) {
    const Vec2& v = b.local[i];
    out[i] = Vec2(p.x + v.x * c - v.y * s, p.y + v.x * s + v.y * c);
  }
}

// Support of a polygon along a direction: the vertex that reaches furthest.
int supportIndex(const std::vector<Vec2>& poly, const Vec2& dir) {
  int best = 0;
  float bestD = Dot(poly[0], dir);
  for (std::size_t i = 1; i < poly.size(); ++i) {
    const float d = Dot(poly[i], dir);
    if (d > bestD) {
      bestD = d;
      best = int(i);
    }
  }
  return best;
}

// One direction of the separating-axis test: the least-penetrating edge normal
// of A against B. Returns the separation; positive means the shapes are apart.
//
// ConvexHull produces vertices wound so that the shoelace area is positive, and
// (edge.y, -edge.x) is the outward normal for that winding. Getting this
// backwards does not crash -- it makes every contact point inward, so pieces
// suck together instead of separating.
float maxSeparation(const std::vector<Vec2>& A, const std::vector<Vec2>& B, int& edgeOut) {
  float best = -1e30f;
  edgeOut = 0;

  const std::size_t n = A.size();
  for (std::size_t i = 0; i < n; ++i) {
    const Vec2& p1 = A[i];
    const Vec2& p2 = A[(i + 1) % n];
    const Vec2 edge = p2 - p1;
    const Vec2 normal = Normalize(Vec2(edge.y, -edge.x));
    if (normal.x == 0.0f && normal.y == 0.0f) continue;  // degenerate edge

    const int s = supportIndex(B, -normal);
    const float sep = Dot(B[std::size_t(s)] - p1, normal);
    if (sep > best) {
      best = sep;
      edgeOut = int(i);
    }
  }
  return best;
}

// Clip a segment to the half-plane dot(n, p) <= offset, keeping at most two
// points. This is the reference-face clip that turns two overlapping polygons
// into the one or two contact points a solver actually needs.
int clipSegment(const Vec2 in[2], const Vec2& n, float offset, Vec2 out[2]) {
  int count = 0;
  const float d0 = Dot(n, in[0]) - offset;
  const float d1 = Dot(n, in[1]) - offset;

  if (d0 <= 0.0f) out[count++] = in[0];
  if (d1 <= 0.0f && count < 2) out[count++] = in[1];

  if (d0 * d1 < 0.0f && count < 2) {
    const float t = d0 / (d0 - d1);
    out[count++] = in[0] + (in[1] - in[0]) * t;
  }
  return count;
}

}  // namespace

// ---------------------------------------------------------------------------

void World::reset(const std::vector<Shard>& shards, const ForceParams& fp,
                  const PhysicsParams& pp, const FrameScale& fs,
                  const std::vector<Collider>& colliders) {
  _params = WorldParams{};
  _params.gravity = GravityVector(pp, fs);
  _params.drag = pp.viscosity * kDragScale;
  _params.restitution = Clamp01(pp.bounciness);
  _params.friction = pp.friction < 0.0f ? 0.0f : pp.friction;
  _params.collideShards = pp.collideShards;
  _params.floorEnabled = pp.floorEnabled;
  _params.floorY = pp.floorY * fs.height;
  _params.wallsEnabled = pp.wallsEnabled;
  _params.leftX = 0.0f;
  _params.rightX = fs.width;
  _params.substeps = std::max(1, std::min(16, pp.substeps));
  _params.frameW = fs.width;
  _params.frameH = fs.height;
  _params.cullMargin = std::max(fs.width, fs.height) * 1.5f;

  _bodies.clear();
  _bodies.resize(shards.size());
  _contacts.clear();
  _time = float(fp.startFrame);

  // Resolved to pixels once, here, rather than per substep. Everything else in
  // the solver works in pixels, and converting inside the contact loop would be
  // the same multiply a few hundred thousand times a frame.
  _colliders.clear();
  _colliders.reserve(colliders.size());
  for (const Collider& c : colliders) {
    Collider p;
    p.x1 = c.x1 * fs.width;
    p.y1 = c.y1 * fs.height;
    p.x2 = c.x2 * fs.width;
    p.y2 = c.y2 * fs.height;
    _colliders.push_back(p);
  }

  for (std::size_t i = 0; i < shards.size(); ++i) {
    const Shard& sh = shards[i];
    Body& b = _bodies[i];

    ShardSim ss;
    ss.cx = sh.cx;
    ss.cy = sh.cy;
    ss.area = sh.area;
    ss.radius = sh.radius;
    ss.inertiaPerMass = sh.inertiaPerMass;

    const uint32_t id = uint32_t(sh.id);
    const float mass = MassOf(ss, fs, pp, id, uint32_t(fp.seed));

    b.rest = Vec2(sh.cx, sh.cy);
    b.radius = sh.radius;
    b.detachFrame = DetachFrame(ss, fp, fs);
    b.pos = Vec2(0.0f, 0.0f);
    b.angle = 0.0f;

    // Hull vertices relative to the centroid, stored once; the solver rotates
    // them per substep rather than keeping a transformed copy in sync.
    if (sh.poly.size() >= 3) {
      b.local.resize(sh.poly.size());
      for (std::size_t v = 0; v < sh.poly.size(); ++v)
        b.local[v] = Vec2(sh.poly[v].x - sh.cx, sh.poly[v].y - sh.cy);
    }

    // Rotational inertia falls back to a disc of the shard's own radius when the
    // hull was degenerate, so a sliver still tumbles rather than spinning
    // instantly to infinity.
    const float inertia =
        mass * (sh.inertiaPerMass > 0.0f ? sh.inertiaPerMass : sh.radius * sh.radius);

    b.invMassFull = 1.0f / std::max(0.0001f, mass);
    b.invInertiaFull = inertia > 1e-6f ? 1.0f / inertia : 0.0f;

    // The forces that will reach this shard, in the order they arrive. They are
    // NOT folded into a launch velocity: a delayed force has to act on a piece
    // that is already moving, which is the whole reason the delay control is
    // interesting. Velocity therefore starts at zero and is built up by
    // impulses as they fall due.
    b.vel = Vec2(0.0f, 0.0f);
    b.w = 0.0f;
    b.pendingCount = GatherImpulses(ss, fp, pp, fs, id, mass, b.pending);
    b.pendingNext = 0;
    b.spinAtDetach = InitialSpin(ss, fp, pp, fs, id, mass);

    // Every piece starts immovable, and stays that way until its detach frame.
    //
    // That is not just an optimisation: a piece that has not let go yet is a
    // wall as far as the rest of the simulation is concerned, which is exactly
    // what makes Inside Radius Only look right -- debris knocked loose by a
    // force bounces off the part of the image still standing, instead of
    // falling straight through it.
    b.invMass = 0.0f;
    b.invInertia = 0.0f;
    b.active = false;
    b.culled = false;
  }
}

void World::releaseDue(float now) {
  for (Body& b : _bodies) {
    if (b.culled) continue;

    // Let go, if it is time. A piece can be released before any force reaches
    // it -- All At Once does exactly that -- in which case it simply starts
    // falling and waits for its shove.
    if (!b.active && now + 1e-4f >= b.detachFrame) {
      b.active = true;
      b.invMass = b.invMassFull;
      b.invInertia = b.invInertiaFull;
      b.w += b.spinAtDetach;
    }
    if (!b.active) continue;

    // Impulses that have come due. Applied to velocity directly rather than
    // through the contact solver: a force is an instantaneous change of motion,
    // not something the solver should be allowed to argue with.
    while (b.pendingNext < b.pendingCount &&
           b.pending[b.pendingNext].time <= now + 1e-4f) {
      b.vel += b.pending[b.pendingNext].linear;
      ++b.pendingNext;
    }
  }
}

void World::integrate(float dt) {
  for (Body& b : _bodies) {
    if (!b.active || b.culled) continue;

    // Exponential drag rather than a subtraction, so a large dt cannot flip the
    // velocity's sign and make heavy damping look like a bounce.
    if (_params.drag > 0.0f) {
      const float decay = expf(-_params.drag * dt);
      b.vel = b.vel * decay + _params.gravity * ((1.0f - decay) / _params.drag);
      b.w *= decay;
    } else {
      b.vel += _params.gravity * dt;
    }

    Vec2 delta = b.vel * dt;

    // Anti-tunnelling backstop: never travel more than half the body's own
    // radius in one substep. This trades a slightly slow piece for one that does
    // not teleport through the floor, and only engages on a shard thrown far
    // harder than the defaults allow.
    const float maxTravel = std::max(2.0f, b.radius * 0.5f);
    const float len = Length(delta);
    if (len > maxTravel) delta = delta * (maxTravel / len);

    b.pos += delta;
    b.angle += b.w * dt;

    // Once a piece is well outside the frame it can never come back, so it stops
    // being simulated. Without this a long clip keeps solving contacts between
    // pieces that left the screen a hundred frames ago.
    const Vec2 p = b.worldPos();
    if (p.x < -_params.cullMargin || p.x > _params.frameW + _params.cullMargin ||
        p.y < -_params.cullMargin || p.y > _params.frameH + _params.cullMargin) {
      b.culled = true;
      b.invMass = 0.0f;
      b.invInertia = 0.0f;
    }
  }
}

void World::collectStaticContacts() {
  if (!_params.floorEnabled && !_params.wallsEnabled) return;

  struct Plane {
    Vec2 n;    ///< unit, points OUT of the solid into free space
    float d;   ///< the solid is where dot(n, p) < d
    bool on;
  };
  const Plane planes[3] = {
      {Vec2(0.0f, -1.0f), -_params.floorY, _params.floorEnabled},
      {Vec2(1.0f, 0.0f), _params.leftX, _params.wallsEnabled},
      {Vec2(-1.0f, 0.0f), -_params.rightX, _params.wallsEnabled},
  };

  for (std::size_t i = 0; i < _bodies.size(); ++i) {
    Body& b = _bodies[i];
    // Only movable bodies need static contacts: an immovable one is already
    // exactly where it belongs and no impulse could change that.
    if (!b.active || b.culled || b.invMass <= 0.0f || b.local.empty()) continue;

    transformHull(b, _worldVerts);

    for (const Plane& pl : planes) {
      if (!pl.on) continue;

      // The deepest vertex becomes the contact. One point per plane per substep
      // is coarse, but a shard landing flat gets its second corner on the next
      // substep and settles within a frame.
      int deepest = -1;
      float sep = 0.0f;
      for (std::size_t v = 0; v < _worldVerts.size(); ++v) {
        const float s = Dot(pl.n, _worldVerts[v]) - pl.d;
        if (s < sep) {
          sep = s;
          deepest = int(v);
        }
      }
      if (deepest < 0) continue;

      Contact c;
      c.a = int(i);
      c.b = -1;
      // Contact normals always point from A toward the other party. For a plane
      // the other party is the solid, which lies opposite the plane's outward
      // normal -- so this is negated. With the sign the other way the solver
      // drives shards INTO the floor, which reads as pieces sinking through it.
      c.normal = -pl.n;
      c.point = _worldVerts[std::size_t(deepest)];
      c.depth = -sep;
      _contacts.push_back(c);
    }
  }
}

void World::collectColliderContacts() {
  if (_colliders.empty()) return;

  // How solid a drawn line is. A collider has no thickness of its own, so
  // without this a shard would come to rest with its corner exactly on the line
  // and read as balancing on nothing.
  constexpr float kHalfThickness = 1.5f;

  for (std::size_t i = 0; i < _bodies.size(); ++i) {
    Body& b = _bodies[i];
    if (!b.active || b.culled || b.invMass <= 0.0f || b.local.empty()) continue;

    transformHull(b, _worldVerts);
    const Vec2 centre = b.worldPos();

    for (const Collider& c : _colliders) {
      const Vec2 a(c.x1, c.y1);
      const Vec2 e(c.x2, c.y2);
      const Vec2 ab = e - a;
      const float len2 = LengthSq(ab);
      if (len2 < 1.0f) continue;  // a click, not a line

      // Cheap reject: the whole body is further from the line than it is wide.
      Vec2 n = Normalize(Vec2(ab.y, -ab.x));
      if (n.x == 0.0f && n.y == 0.0f) continue;

      // A drawn line is solid from BOTH sides, so which way it pushes depends on
      // which side the shard came from. Orienting the normal toward the body's
      // centre each substep is what gives that, and it is why a collider can be
      // landed on from above and bounced off from below without being two
      // different objects.
      const float side = Dot(centre - a, n);
      if (side < 0.0f) n = -n;

      if (fabsf(side) > b.radius + kHalfThickness) continue;

      // The deepest vertex that is actually alongside the segment. Vertices past
      // either end are ignored, which is what makes the collider finite -- a
      // shard sliding off the end of a ramp keeps going instead of catching on
      // an invisible infinite wall.
      int deepest = -1;
      float depth = 0.0f;
      for (std::size_t v = 0; v < _worldVerts.size(); ++v) {
        const Vec2 rel = _worldVerts[v] - a;
        const float t = Dot(rel, ab) / len2;
        if (t < 0.0f || t > 1.0f) continue;

        const float d = kHalfThickness - Dot(rel, n);  // > 0 means inside
        if (d > depth) {
          depth = d;
          deepest = int(v);
        }
      }
      if (deepest < 0) continue;

      Contact ct;
      ct.a = int(i);
      ct.b = -1;
      // From A toward the obstacle, matching the convention the solver and the
      // floor contacts both use.
      ct.normal = -n;
      ct.point = _worldVerts[std::size_t(deepest)];
      ct.depth = depth;
      _contacts.push_back(ct);
    }
  }
}

void World::collectShardContacts() {
  if (!_params.collideShards) return;

  // Uniform-grid broadphase. Cell size tracks the mean body radius, so a frame
  // of confetti and a frame of four big slabs both end up with a handful of
  // candidates per cell.
  //
  // Immovable bodies are included: pieces that have not detached yet are
  // obstacles, and leaving them out is what would let debris fall through the
  // standing part of the image.
  auto eligible = [](const Body& b) { return !b.culled && !b.local.empty(); };

  float radiusSum = 0.0f;
  int counted = 0;
  float minX = 1e30f, minY = 1e30f, maxX = -1e30f, maxY = -1e30f;
  for (const Body& b : _bodies) {
    if (!eligible(b)) continue;
    radiusSum += b.radius;
    ++counted;
    const Vec2 p = b.worldPos();
    minX = std::min(minX, p.x - b.radius);
    minY = std::min(minY, p.y - b.radius);
    maxX = std::max(maxX, p.x + b.radius);
    maxY = std::max(maxY, p.y + b.radius);
  }
  if (counted < 2) return;

  _cellSize = std::max(8.0f, 2.0f * radiusSum / float(counted));
  _gridMinX = minX;
  _gridMinY = minY;
  _gridW = std::max(1, std::min(2048, int((maxX - minX) / _cellSize) + 1));
  _gridH = std::max(1, std::min(2048, int((maxY - minY) / _cellSize) + 1));

  const int nc = _gridW * _gridH;
  _cellStart.assign(std::size_t(nc) + 1, 0);

  auto cellOf = [&](const Vec2& p, int& cx, int& cy) {
    cx = std::max(0, std::min(_gridW - 1, int((p.x - _gridMinX) / _cellSize)));
    cy = std::max(0, std::min(_gridH - 1, int((p.y - _gridMinY) / _cellSize)));
  };

  // Count, prefix, fill -- CSR, in body order, so the contact list comes out in
  // the same sequence on every run. Traversal order is the whole determinism
  // argument, so it must not depend on anything but the index.
  for (const Body& b : _bodies) {
    if (!eligible(b)) continue;
    int cx, cy;
    cellOf(b.worldPos(), cx, cy);
    ++_cellStart[std::size_t(cy * _gridW + cx) + 1];
  }
  for (int i = 0; i < nc; ++i) _cellStart[std::size_t(i) + 1] += _cellStart[std::size_t(i)];
  _cellItems.resize(std::size_t(_cellStart[std::size_t(nc)]));

  std::vector<int> cursor(_cellStart.begin(), _cellStart.end() - 1);
  for (std::size_t i = 0; i < _bodies.size(); ++i) {
    if (!eligible(_bodies[i])) continue;
    int cx, cy;
    cellOf(_bodies[i].worldPos(), cx, cy);
    _cellItems[std::size_t(cursor[std::size_t(cy * _gridW + cx)]++)] = int(i);
  }

  std::vector<Vec2> hullA, hullB;

  for (std::size_t ia = 0; ia < _bodies.size(); ++ia) {
    const Body& A = _bodies[ia];
    if (!eligible(A)) continue;

    int cx, cy;
    cellOf(A.worldPos(), cx, cy);

    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        const int nx = cx + dx, ny = cy + dy;
        if (nx < 0 || ny < 0 || nx >= _gridW || ny >= _gridH) continue;

        const int cell = ny * _gridW + nx;
        for (int k = _cellStart[std::size_t(cell)]; k < _cellStart[std::size_t(cell) + 1]; ++k) {
          const int ib = _cellItems[std::size_t(k)];
          // Each unordered pair once, always with the lower index as A, so the
          // normal's sign never depends on traversal order.
          if (ib <= int(ia)) continue;

          const Body& B = _bodies[std::size_t(ib)];
          // Two immovable bodies cannot resolve anything between them.
          if (A.invMass <= 0.0f && B.invMass <= 0.0f) continue;

          // Cheap radius reject before the SAT.
          const Vec2 d = B.worldPos() - A.worldPos();
          const float rr = A.radius + B.radius;
          if (LengthSq(d) > rr * rr) continue;

          transformHull(A, hullA);
          transformHull(B, hullB);

          int edgeA = 0, edgeB = 0;
          const float sepA = maxSeparation(hullA, hullB, edgeA);
          if (sepA > 0.0f) continue;
          const float sepB = maxSeparation(hullB, hullA, edgeB);
          if (sepB > 0.0f) continue;

          // The reference face is the one with the LEAST penetration. Using the
          // other one puts contact points on the wrong side, and pieces stick
          // together instead of pushing apart.
          const bool useA = sepA >= sepB;
          const std::vector<Vec2>& ref = useA ? hullA : hullB;
          const std::vector<Vec2>& inc = useA ? hullB : hullA;
          const int refEdge = useA ? edgeA : edgeB;

          const std::size_t rn = ref.size();
          const Vec2 r1 = ref[std::size_t(refEdge)];
          const Vec2 r2 = ref[(std::size_t(refEdge) + 1) % rn];
          const Vec2 refDir = Normalize(r2 - r1);
          if (refDir.x == 0.0f && refDir.y == 0.0f) continue;
          const Vec2 refNormal(refDir.y, -refDir.x);

          // Incident edge: the one on the other body most anti-parallel to the
          // reference normal.
          std::size_t incEdge = 0;
          float minDot = 1e30f;
          const std::size_t in = inc.size();
          for (std::size_t e = 0; e < in; ++e) {
            const Vec2 ev = inc[(e + 1) % in] - inc[e];
            const Vec2 en = Normalize(Vec2(ev.y, -ev.x));
            const float dt = Dot(en, refNormal);
            if (dt < minDot) {
              minDot = dt;
              incEdge = e;
            }
          }

          Vec2 seg[2] = {inc[incEdge], inc[(incEdge + 1) % in]};
          Vec2 clipped[2];
          if (clipSegment(seg, -refDir, Dot(-refDir, r1), clipped) < 2) continue;
          seg[0] = clipped[0];
          seg[1] = clipped[1];
          if (clipSegment(seg, refDir, Dot(refDir, r2), clipped) < 2) continue;

          const float refOffset = Dot(refNormal, r1);
          for (int p = 0; p < 2; ++p) {
            const float sep = Dot(refNormal, clipped[p]) - refOffset;
            if (sep > 0.0f) continue;

            Contact c;
            c.a = int(ia);
            c.b = ib;
            // Always from A toward B, whichever body owned the reference face.
            c.normal = useA ? refNormal : -refNormal;
            c.point = clipped[p];
            c.depth = -sep;
            _contacts.push_back(c);
          }
        }
      }
    }
  }
}

void World::solveContacts(float dt) {
  if (_contacts.empty()) return;

  // A is pushed along -normal, B along +normal, because the normal points from
  // A toward B.
  auto applyImpulse = [&](int idx, const Vec2& imp, const Vec2& r, float sign) {
    if (idx < 0) return;
    Body& b = _bodies[std::size_t(idx)];
    if (b.invMass <= 0.0f) return;
    b.vel += imp * (sign * b.invMass);
    b.w += sign * b.invInertia * Cross(r, imp);
  };

  for (int iter = 0; iter < kVelocityIterations; ++iter) {
    for (Contact& c : _contacts) {
      Body& A = _bodies[std::size_t(c.a)];
      Body* B = c.b >= 0 ? &_bodies[std::size_t(c.b)] : nullptr;

      const Vec2 ra = c.point - A.worldPos();
      const Vec2 rb = B ? c.point - B->worldPos() : Vec2(0.0f, 0.0f);

      auto pointVel = [](const Body& b, const Vec2& r) {
        return b.vel + Vec2(-b.w * r.y, b.w * r.x);
      };

      const Vec2 vA = pointVel(A, ra);
      const Vec2 vB = B ? pointVel(*B, rb) : Vec2(0.0f, 0.0f);
      const float vn = Dot(vB - vA, c.normal);

      float kn = A.invMass + A.invInertia * Cross(ra, c.normal) * Cross(ra, c.normal);
      if (B) kn += B->invMass + B->invInertia * Cross(rb, c.normal) * Cross(rb, c.normal);
      if (kn <= 1e-12f) continue;

      // Positional bias, with a slop so resting contacts stop correcting.
      const float bias = kBaumgarte * std::max(0.0f, c.depth - kSlop) / dt;
      const float rest = vn < -kRestingSpeed ? -_params.restitution * vn : 0.0f;

      float lambda = (-vn + bias + rest) / kn;

      // Accumulated-impulse clamping: the running total may never pull, though
      // an individual iteration's increment may be negative.
      const float oldN = c.normalImpulse;
      c.normalImpulse = std::max(0.0f, oldN + lambda);
      lambda = c.normalImpulse - oldN;

      const Vec2 imp = c.normal * lambda;
      applyImpulse(c.a, imp, ra, -1.0f);
      if (B) applyImpulse(c.b, imp, rb, +1.0f);

      // Friction along the tangent, bounded by Coulomb's cone. Recomputed after
      // the normal impulse so it sees the corrected velocity.
      if (_params.friction > 0.0f) {
        const Vec2 tangent(-c.normal.y, c.normal.x);
        const Vec2 vA2 = pointVel(A, ra);
        const Vec2 vB2 = B ? pointVel(*B, rb) : Vec2(0.0f, 0.0f);
        const float vt = Dot(vB2 - vA2, tangent);

        float kt = A.invMass + A.invInertia * Cross(ra, tangent) * Cross(ra, tangent);
        if (B) kt += B->invMass + B->invInertia * Cross(rb, tangent) * Cross(rb, tangent);
        if (kt <= 1e-12f) continue;

        float lt = -vt / kt;
        const float maxT = _params.friction * c.normalImpulse;
        const float oldT = c.tangentImpulse;
        c.tangentImpulse = std::max(-maxT, std::min(maxT, oldT + lt));
        lt = c.tangentImpulse - oldT;

        const Vec2 timp = tangent * lt;
        applyImpulse(c.a, timp, ra, -1.0f);
        if (B) applyImpulse(c.b, timp, rb, +1.0f);
      }
    }
  }
}

void World::step(float frame) {
  const int n = _params.substeps;
  const float dt = 1.0f / float(n);

  // The frame being stepped into ends at `frame`, so the substeps run over
  // (frame-1, frame]. Releases and impulses are resolved at substep resolution
  // rather than once per frame, so a delay of 10.5 frames lands where it says
  // it does instead of snapping to the frame boundary.
  _time = frame - 1.0f;

  for (int s = 0; s < n; ++s) {
    _time += dt;
    releaseDue(_time);

    integrate(dt);

    _contacts.clear();
    collectStaticContacts();
    collectColliderContacts();
    collectShardContacts();
    solveContacts(dt);
  }
  _time = frame;
}

}  // namespace phys
}  // namespace shp
