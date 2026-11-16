// Ambient 3D backdrop: a neural-network brain that reacts to mouse and scroll. Runs in the page, behind the Streamlit content.
(function () {
  var d = document;
  if (d.getElementById('ns-bg')) return;
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var cv = d.createElement('canvas'); cv.id = 'ns-bg';
  cv.style.cssText = 'position:fixed;inset:0;width:100vw;height:100vh;z-index:0;pointer-events:none;opacity:0;transition:opacity 1.6s ease';
  d.body.appendChild(cv);
  var renderer = new THREE.WebGLRenderer({ canvas: cv, antialias: true, alpha: true });
  renderer.setPixelRatio(1); renderer.setClearColor(0x000000, 0);
  var scene = new THREE.Scene(), cam = new THREE.PerspectiveCamera(40, 1, 0.1, 100); cam.position.set(0, 0, 6.2);
  var grp = new THREE.Group(); scene.add(grp);
  // deterministic pseudo random
  var seed = 7; function rnd() { seed = (seed * 16807) % 2147483647; return (seed - 1) / 2147483646; }
  var N = 520, P = [], i, j;
  for (i = 0; i < N; i++) {
    var u = rnd() * 2 - 1, th = rnd() * Math.PI * 2, r = Math.cbrt(rnd());
    var s = Math.sqrt(1 - u * u), x = s * Math.cos(th) * r, y = u * r * 0.82, z = s * Math.sin(th) * r * 0.78;
    // surface bias so it reads as a brain outline
    var k = 0.8 + 0.2 * Math.pow(rnd(), 0.5); x *= k; y *= k; z *= k;
    // gyri ripple on the outer shell and a longitudinal fissure between hemispheres
    var ripple = 1 + 0.09 * Math.sin(x * 11 + z * 3) * Math.sin(y * 9 + z * 6);
    x *= ripple; y *= ripple; z *= ripple;
    x += (x >= 0 ? 0.045 : -0.045);
    P.push([x * 2.1, y * 1.75 - (y < -0.55 ? 0.0 : 0), z * 1.5]);
  }
  // edges between near neighbours
  var seg = [], edges = [], maxD = 0.5, maxE = 3;
  var deg = new Array(N).fill(0);
  for (i = 0; i < N; i++) for (j = i + 1; j < N; j++) {
    var dx = P[i][0] - P[j][0], dy = P[i][1] - P[j][1], dz = P[i][2] - P[j][2], dd = dx * dx + dy * dy + dz * dz;
    if (dd < maxD * maxD && deg[i] < maxE + 1 && deg[j] < maxE + 1) { seg.push(P[i][0], P[i][1], P[i][2], P[j][0], P[j][1], P[j][2]); edges.push([i, j]); deg[i]++; deg[j]++; }
  }
  var adj = []; for (i = 0; i < N; i++) adj.push([]); edges.forEach(function (e, k) { adj[e[0]].push(k); adj[e[1]].push(k); });
  var lg = new THREE.BufferGeometry(); lg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(seg), 3));
  grp.add(new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0x9aa0a8, transparent: true, opacity: 0.16, depthWrite: false })));
  var np = new Float32Array(N * 3), nc = new Float32Array(N * 3);
  for (i = 0; i < N; i++) {
    np[3 * i] = P[i][0]; np[3 * i + 1] = P[i][1]; np[3 * i + 2] = P[i][2];
    var hot = rnd() < 0.07; nc[3 * i] = hot ? 0.96 : 0.82; nc[3 * i + 1] = hot ? 0.77 : 0.84; nc[3 * i + 2] = hot ? 0.09 : 0.86;
  }
  var ng = new THREE.BufferGeometry(); ng.setAttribute('position', new THREE.BufferAttribute(np, 3)); ng.setAttribute('color', new THREE.BufferAttribute(nc, 3));
  function tex() { var c = d.createElement('canvas'); c.width = c.height = 32; var g = c.getContext('2d'), r = g.createRadialGradient(16, 16, 0, 16, 16, 16); r.addColorStop(0, 'rgba(255,255,255,1)'); r.addColorStop(0.5, 'rgba(255,255,255,.4)'); r.addColorStop(1, 'rgba(255,255,255,0)'); g.fillStyle = r; g.fillRect(0, 0, 32, 32); return new THREE.CanvasTexture(c); }
  var T = tex();
  grp.add(new THREE.Points(ng, new THREE.PointsMaterial({ size: 0.075, map: T, vertexColors: true, transparent: true, opacity: 0.7, depthWrite: false, blending: THREE.AdditiveBlending })));
  // travelling signals
  var S = 36, sp = new Float32Array(S * 3), st = [];
  for (i = 0; i < S; i++) st.push({ e: Math.floor(rnd() * edges.length), t: rnd(), v: 0.004 + rnd() * 0.01 });
  var sg = new THREE.BufferGeometry(); sg.setAttribute('position', new THREE.BufferAttribute(sp, 3));
  grp.add(new THREE.Points(sg, new THREE.PointsMaterial({ size: 0.15, map: T, color: 0xf5c518, transparent: true, opacity: 0.95, depthWrite: false, blending: THREE.AdditiveBlending })));
  function stepSignals() {
    for (var q = 0; q < S; q++) {
      var o = st[q]; o.t += o.v; if (o.t >= 1) { // hop to a connected edge
        var node = edges[o.e][1], cand = adj[node];
        o.e = cand.length ? cand[Math.floor(rnd() * cand.length)] : Math.floor(rnd() * edges.length); o.t = 0;
      }
      var a = P[edges[o.e][0]], b = P[edges[o.e][1]];
      sp[3 * q] = a[0] + (b[0] - a[0]) * o.t; sp[3 * q + 1] = a[1] + (b[1] - a[1]) * o.t; sp[3 * q + 2] = a[2] + (b[2] - a[2]) * o.t;
    }
    sg.attributes.position.needsUpdate = true;
  }
  var mx = 0, my = 0, tx = 0, ty = 0, sy = 0;
  window.addEventListener('mousemove', function (e) { tx = e.clientX / window.innerWidth - 0.5; ty = e.clientY / window.innerHeight - 0.5; }, { passive: true });
  function scrollEl() { return d.querySelector('section.main, [data-testid="stMain"], [data-testid="stAppViewContainer"]'); }
  function resize() { var w = window.innerWidth, h = window.innerHeight; renderer.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix(); grp.position.x = w > 900 ? 1.55 : 0; grp.scale.setScalar(w > 900 ? 1 : 0.8); }
  window.addEventListener('resize', resize); resize();
  var t0 = performance.now(); cv.style.opacity = '0.9';
  var last = 0;
  function frame(now) {
    requestAnimationFrame(frame);
    if (now - last < 33) return;   // about 30 frames per second is plenty for an ambient background
    last = now;
    if (!d.hidden) {
      var el = scrollEl(); sy = el ? el.scrollTop : window.scrollY;
      mx += (tx - mx) * 0.05; my += (ty - my) * 0.05;
      var t = (performance.now() - t0) / 1000;
      grp.rotation.y = (reduce ? 0.5 : t * 0.06) + mx * 0.9 + sy * 0.0007;
      grp.rotation.x = my * 0.35 - 0.1 + Math.sin(sy * 0.0012) * 0.12;
      if (!reduce) stepSignals();
      renderer.render(scene, cam);
    }
  }
  stepSignals(); requestAnimationFrame(frame);
})();
