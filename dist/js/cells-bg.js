/* scCopilot 微观生命 3D 背景
 *
 * 纯装饰层：不读数据、不参与任何数字的生成。
 * - 真核细胞：不规则膜 + 核膜/核仁/核孔 + 线粒体（含嵴）+ 高尔基 + 内质网 + 细胞骨架 + 囊泡；
 * - DNA：B 型双螺旋（螺距 10.5bp/圈），磷酸骨架为连续管，碱基对 A–T 两根 / G–C 三根氢键；
 * - 蛋白质：α 螺旋飘带 + β 折叠箭头 + 无规线团 loop（结构生物学 ribbon 画法）；
 * - 微生物：杆菌（周生鞭毛）+ 球菌链；绿藻（衣藻：杯状叶绿体 + 红色眼点 + 双鞭毛脉冲游动）；
 * - 萤火虫：鞘翅 + 前胸背板 + 发光器，呼吸式闪烁，趋光——鼠标越近飞得越近、闪得越亮；
 * - 气氛层：双层反向漂移的荧光尘埃 + 上浮的荧光孢子 + 焦外光斑 + 极淡的远处晶格，
 *   和 CSS 的 #sc-aurora / #sc-grid / #sc-grain 一起把深蓝铺满；掉帧时先关这一层；
 * - 青→蓝→紫生物荧光缓慢循环；prefers-reduced-motion 只渲染一帧；WebGL 不可用静默退回 CSS 渐变。
 *
 * 造型按"显微镜/模型图的可读轮廓"取舍，不追求分子级精确：装饰层不能暗示任何实验结论。
 */
(function () {
  'use strict';

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function init() {
    if (typeof THREE === 'undefined') return;
    const canvas = document.getElementById('cell-bg');
    if (!canvas) return;

    const renderer = new THREE.WebGLRenderer({
      canvas, alpha: true, antialias: true, powerPreference: 'low-power',
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 200);
    camera.position.set(0, 0, 26);

    scene.add(new THREE.AmbientLight(0x8fb8ff, 0.6));
    const keyLight = new THREE.PointLight(0x66ccff, 1.4, 120);
    keyLight.position.set(14, 10, 18);
    scene.add(keyLight);
    const rimLight = new THREE.PointLight(0xa78bfa, 1.0, 120);
    rimLight.position.set(-16, -8, 10);
    scene.add(rimLight);
    // 只给一只最亮的萤火虫配真光源，其余靠 sprite 辉光，避免一堆实时灯拖帧
    const lanternLight = new THREE.PointLight(0xffc861, 0, 30);
    scene.add(lanternLight);

    function glowTexture(stops) {
      const c = document.createElement('canvas');
      c.width = c.height = 128;
      const ctx = c.getContext('2d');
      const g = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
      for (const [pos, col] of stops) g.addColorStop(pos, col);
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, 128, 128);
      return new THREE.CanvasTexture(c);
    }
    const glowMap = glowTexture([
      [0, 'rgba(255,255,255,0.85)'], [0.25, 'rgba(255,255,255,0.32)'],
      [0.6, 'rgba(255,255,255,0.08)'], [1, 'rgba(255,255,255,0)'],
    ]);
    const haloMap = glowTexture([
      [0, 'rgba(255,255,255,0.95)'], [0.12, 'rgba(255,255,255,0.55)'],
      [0.45, 'rgba(255,255,255,0.12)'], [1, 'rgba(255,255,255,0)'],
    ]);

    // 可复现随机：背景布局每次访问一致，避免"每次打开细胞位置乱跳"。
    let seed = 42;
    function rand() {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 4294967296;
    }
    const between = (a, b) => a + (b - a) * rand();
    const pick = (arr) => arr[Math.floor(rand() * arr.length) % arr.length];

    // 屏幕左半边是 hero 正文：把大物件推到更深的 z，远处仍看得见，但不会在文字上打亮块
    function settle(p, depth) {
      if (p.x < -1.5 && p.z > -9) p.z -= depth;
      return p;
    }

    // 沿球面撒点（均匀方向 + 半径），用于细胞器定位
    function shellPoints(n, radius, inner) {
      const pts = [];
      for (let i = 0; i < n; i++) {
        const theta = rand() * Math.PI * 2;
        const phi = Math.acos(2 * rand() - 1);
        const rr = radius * (inner + (1 - inner) * Math.cbrt(rand()));
        pts.push(new THREE.Vector3(
          rr * Math.sin(phi) * Math.cos(theta),
          rr * Math.sin(phi) * Math.sin(theta),
          rr * Math.cos(phi)
        ));
      }
      return pts;
    }

    // 光滑的低频扰动：把规则球变成"有张力但不圆"的细胞轮廓
    function blobby(radius, a1, a2, freq) {
      const geo = new THREE.SphereGeometry(radius, 40, 28);
      const p = geo.attributes.position;
      const v = new THREE.Vector3();
      for (let i = 0; i < p.count; i++) {
        v.fromBufferAttribute(p, i);
        const d = v.clone().normalize();
        const k = a1 * Math.sin(freq * d.x * Math.PI + d.y * 2.1)
                + a2 * Math.cos(freq * 0.8 * d.y * Math.PI - d.z * 1.7);
        v.addScaledVector(d, radius * k);
        p.setXYZ(i, v.x, v.y, v.z);
      }
      geo.computeVertexNormals();
      return geo;
    }

    /* ---------------------------------------------------------- 真核细胞 */
    const cells = [];
    const CELL_COUNT = 6;
    for (let i = 0; i < CELL_COUNT; i++) {
      const r = between(1.7, 2.9);
      const group = new THREE.Group();
      const hueBase = 0.5 + (i % 4) * 0.05;
      const color = new THREE.Color().setHSL(hueBase, 0.8, 0.6);

      const membrane = new THREE.Mesh(
        blobby(r, 0.055, 0.04, 2.4),
        new THREE.MeshPhongMaterial({
          color, emissive: color.clone().multiplyScalar(0.28),
          transparent: true, opacity: 0.13, shininess: 80, depthWrite: false,
        })
      );
      group.add(membrane);

      // 核：核膜 + 核仁 + 核孔（小暗点）
      const nucR = r * 0.36;
      const nucleus = new THREE.Group();
      const envelope = new THREE.Mesh(
        blobby(nucR, 0.04, 0.03, 3),
        new THREE.MeshPhongMaterial({
          color: color.clone().lerp(new THREE.Color(0xffffff), 0.35),
          emissive: color.clone().multiplyScalar(0.5),
          transparent: true, opacity: 0.5, depthWrite: false,
        })
      );
      nucleus.add(envelope);
      nucleus.add(new THREE.Mesh(
        new THREE.SphereGeometry(nucR * 0.34, 16, 12),
        new THREE.MeshPhongMaterial({
          color: color.clone().multiplyScalar(1.25), transparent: true, opacity: 0.9,
        })
      ));
      nucleus.children[1].position.set(nucR * 0.25, -nucR * 0.15, nucR * 0.2);
      const pores = shellPoints(22, nucR * 1.01, 1);
      const poreGeo = new THREE.BufferGeometry().setFromPoints(pores);
      nucleus.add(new THREE.Points(poreGeo, new THREE.PointsMaterial({
        color: 0x08111f, size: nucR * 0.12, transparent: true, opacity: 0.55,
      })));
      nucleus.position.set(r * 0.12, r * 0.05, 0);
      group.add(nucleus);

      // 线粒体：短杆 + 内部嵴（半圈环），最容易被认出的细胞器
      const mitoMat = new THREE.MeshPhongMaterial({
        color: color.clone().lerp(new THREE.Color(0xff9d5c), 0.55),
        emissive: new THREE.Color(0x3a1d0c), transparent: true, opacity: 0.72, depthWrite: false,
      });
      const cristaMat = new THREE.MeshBasicMaterial({
        color: 0xffd9a8, transparent: true, opacity: 0.45, depthWrite: false,
      });
      const mitoProto = new THREE.Group();
      const mitoLen = r * 0.34;
      mitoProto.add(new THREE.Mesh(new THREE.SphereGeometry(mitoLen, 18, 12), mitoMat));
      mitoProto.children[0].scale.set(1, 0.42, 0.42);
      for (let k = -1; k <= 1; k++) {
        const ring = new THREE.Mesh(new THREE.TorusGeometry(mitoLen * 0.3, mitoLen * 0.05, 6, 14, Math.PI * 1.5), cristaMat);
        ring.rotation.y = Math.PI / 2;
        ring.rotation.x = k % 2 ? 0.4 : -0.4;
        ring.position.x = k * mitoLen * 0.34;
        mitoProto.add(ring);
      }
      const mitos = [];
      for (const m of shellPoints(between(3, 5), r * 0.72, 0.25)) {
        const inst = mitoProto.clone();
        inst.position.copy(m);
        inst.rotation.set(between(-1, 1), between(-1, 1), between(-1, 1));
        group.add(inst);
        mitos.push(inst);
      }

      // 高尔基：同心的扁半环堆叠
      const golgi = new THREE.Group();
      const golgiMat = new THREE.MeshPhongMaterial({
        color: color.clone().lerp(new THREE.Color(0xf0abfc), 0.5),
        transparent: true, opacity: 0.5, depthWrite: false,
      });
      for (let k = 0; k < 3; k++) {
        // 高尔基体是"一摞压扁的囊"，用环会看成甜甜圈
        const c = new THREE.Mesh(new THREE.SphereGeometry(r * (0.22 + k * 0.05), 20, 10), golgiMat);
        c.scale.set(1, 0.08, 0.62);
        c.position.set(-k * r * 0.03, k * r * 0.06, 0);
        golgi.add(c);
      }
      golgi.position.set(-r * 0.55, -r * 0.35, r * 0.2);
      group.add(golgi);

      // 内质网：核膜外延伸的波浪管
      const erPts = [];
      for (let k = 0; k < 3; k++) {
        const curve = [];
        const a0 = between(0, Math.PI * 2);
        for (let s = 0; s <= 16; s++) {
          const t = s / 16;
          const rad = r * (0.45 + t * 0.42);
          const a = a0 + t * between(2.2, 3.4);
          curve.push(new THREE.Vector3(
            Math.cos(a) * rad, Math.sin(a) * rad * 0.8 + Math.sin(t * 9 + k) * r * 0.09,
            Math.sin(a * 0.7) * rad * 0.4
          ));
        }
        erPts.push(new THREE.CatmullRomCurve3(curve));
      }
      for (const cur of erPts) {
        group.add(new THREE.Mesh(
          new THREE.TubeGeometry(cur, 60, r * 0.02, 6, false),
          new THREE.MeshPhongMaterial({
            color: color.clone().lerp(new THREE.Color(0xffffff), 0.2),
            transparent: true, opacity: 0.3, depthWrite: false,
          })
        ));
      }

      // 细胞骨架：贴膜的弧线，只作质感用，压得很淡，否则看成线框球
      for (let k = 0; k < 3; k++) {
        const arc = [];
        const a0 = between(0, Math.PI * 2), b0 = between(-1, 1);
        for (let s = 0; s <= 22; s++) {
          const u = (s / 22 - 0.5) * 2;
          const a = a0 + u * 1.9;
          arc.push(new THREE.Vector3(
            Math.cos(a) * r * 0.97 * Math.cos(b0 + u * 0.5),
            Math.sin(a) * r * 0.97,
            Math.sin(b0 + u * 0.6) * r * 0.9
          ));
        }
        group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(arc),
          new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.22 })));
      }

      // 囊泡 / 核糖体：光点
      const vesGeo = new THREE.BufferGeometry().setFromPoints(shellPoints(34, r * 0.88, 0.2));
      const vesicles = new THREE.Points(vesGeo, new THREE.PointsMaterial({
        color, size: r * 0.075, transparent: true, opacity: 0.8,
        blending: THREE.AdditiveBlending, depthWrite: false,
      }));
      group.add(vesicles);

      const halo = new THREE.Sprite(new THREE.SpriteMaterial({
        map: haloMap, color, transparent: true, opacity: 0.28,
        blending: THREE.AdditiveBlending, depthWrite: false,
      }));
      halo.scale.setScalar(r * 4.6);
      group.add(halo);

      const base = settle(new THREE.Vector3(between(-19, 19), between(-9, 9), between(-12, 1)), 9);
      // 别把 hero 文字区（屏幕中心偏左）堵死
      if (Math.abs(base.x) < 6 && base.y > -2 && base.z > -4) base.x += base.x > 0 ? 9 : -9;
      group.position.copy(base);
      scene.add(group);
      cells.push({
        group, membrane, nucleus, envelope, vesicles, halo, mitos, golgi,
        base, hueBase, phase: rand() * Math.PI * 2,
        drift: between(0.25, 0.55), spin: between(0.18, 0.45) * (rand() > 0.5 ? 1 : -1),
        push: new THREE.Vector3(),
      });
    }

    /* ---------------------------------------------------------- DNA 双螺旋 */
    // B 型：一圈 10.5bp，螺距约 3.4nm。骨架走软管，碱基对按 A–T/G–C 画 2/3 根氢键。
    function makeDNA() {
      const group = new THREE.Group();
      const bpPerTurn = 10.5, bpCount = 46, rise = 0.19;
      const radius = 1.0, half = (bpCount * rise) / 2;
      const backA = [], backB = [], pairs = [];
      for (let i = 0; i < bpCount; i++) {
        const ang = (i / bpPerTurn) * Math.PI * 2;
        const y = -half + i * rise;
        const a = new THREE.Vector3(Math.cos(ang) * radius, y, Math.sin(ang) * radius);
        const b = new THREE.Vector3(Math.cos(ang + Math.PI) * radius, y, Math.sin(ang + Math.PI) * radius);
        backA.push(a); backB.push(b);
        pairs.push({ a, b, gc: rand() > 0.5 });
      }
      const backboneMat = (hex) => new THREE.MeshPhongMaterial({
        color: hex, emissive: new THREE.Color(hex).multiplyScalar(0.4),
        transparent: true, opacity: 0.85, shininess: 60, depthWrite: false,
      });
      for (const [pts, hex] of [[backA, 0x67e8f9], [backB, 0xc4b5fd]]) {
        const cur = new THREE.CatmullRomCurve3(pts);
        group.add(new THREE.Mesh(new THREE.TubeGeometry(cur, 240, 0.075, 7, false), backboneMat(hex)));
      }
      // 碱基：从中缝向两端的短棒；氢键数按 A–T 两根 / G–C 三根画成中缝上的光点
      const rodGeo = new THREE.CylinderGeometry(0.055, 0.055, 1, 6);
      const pairAT = new THREE.MeshPhongMaterial({ color: 0xfde68a, emissive: 0x4a3c0c, transparent: true, opacity: 0.8, depthWrite: false });
      const pairGC = new THREE.MeshPhongMaterial({ color: 0x7dd3fc, emissive: 0x0d3350, transparent: true, opacity: 0.85, depthWrite: false });
      const at = new THREE.InstancedMesh(rodGeo, pairAT, bpCount * 2);
      const gc = new THREE.InstancedMesh(rodGeo, pairGC, bpCount * 2);
      const hbPos = [], hbCol = [];
      const mtx = new THREE.Matrix4(), q = new THREE.Quaternion(), up = new THREE.Vector3(0, 1, 0);
      const axis = new THREE.Vector3(), center = new THREE.Vector3(), scl = new THREE.Vector3();
      const cAT = new THREE.Color(0xfde68a), cGC = new THREE.Color(0x7dd3fc);
      let nAT = 0, nGC = 0;
      for (const p of pairs) {
        axis.copy(p.b).sub(p.a);
        const len = axis.length();
        const dirN = axis.clone().normalize();
        const rodLen = len * 0.44;                 // 两端各留一点，中缝就是氢键的位置
        scl.set(1, rodLen, 1);
        q.setFromUnitVectors(up, dirN);
        for (const end of [p.a, p.b]) {
          center.copy(end).addScaledVector(dirN, end === p.a ? rodLen * 0.5 : -rodLen * 0.5);
          mtx.compose(center, q, scl);
          if (p.gc) gc.setMatrixAt(nGC++, mtx); else at.setMatrixAt(nAT++, mtx);
        }
        const mid = p.a.clone().addScaledVector(dirN, len * 0.5);
        const nBonds = p.gc ? 3 : 2;
        for (let k = 0; k < nBonds; k++) {
          // 氢键画在两碱基之间的中缝上，所以沿配对轴排开，而不是沿螺旋轴
          const off = (k - (nBonds - 1) / 2) * 0.06;
          hbPos.push(mid.x + dirN.x * off, mid.y + dirN.y * off, mid.z + dirN.z * off);
          hbCol.push(...(p.gc ? cGC : cAT).toArray());
        }
      }
      at.count = nAT; gc.count = nGC;
      at.instanceMatrix.needsUpdate = true; gc.instanceMatrix.needsUpdate = true;
      group.add(at, gc);
      const hbGeo = new THREE.BufferGeometry();
      hbGeo.setAttribute('position', new THREE.Float32BufferAttribute(hbPos, 3));
      hbGeo.setAttribute('color', new THREE.Float32BufferAttribute(hbCol, 3));
      group.add(new THREE.Points(hbGeo, new THREE.PointsMaterial({
        size: 0.1, vertexColors: true, transparent: true, opacity: 0.95,
        blending: THREE.AdditiveBlending, depthWrite: false,
      })));
      // 磷酸基团：骨架上的发光节点
      const nodes = new THREE.BufferGeometry().setFromPoints(backA.concat(backB));
      group.add(new THREE.Points(nodes, new THREE.PointsMaterial({
        color: 0xa5f3fc, size: 0.13, transparent: true, opacity: 0.9,
        blending: THREE.AdditiveBlending, depthWrite: false,
      })));
      return group;
    }
    const dna = makeDNA();
    settle(dna.position.set(-14, 1.5, -4.5), 8);
    dna.rotation.z = 0.5;
    scene.add(dna);
    const dna2 = makeDNA();
    dna2.position.set(15.5, -6, -9);
    dna2.rotation.set(0.9, 0, -0.35);
    dna2.scale.setScalar(0.72);
    scene.add(dna2);

    /* ---------------------------------------------------------- 蛋白质 ribbon */
    // 二级结构序列：α 螺旋画成盘起来的管，β 折叠画成扁平箭头，loop 画成细管。
    function makeProtein(hex) {
      const group = new THREE.Group();
      const color = new THREE.Color(hex);
      const mat = new THREE.MeshPhongMaterial({
        color, emissive: color.clone().multiplyScalar(0.45),
        transparent: true, opacity: 0.8, shininess: 70, depthWrite: false,
        side: THREE.DoubleSide,
      });
      const arrowMat = new THREE.MeshBasicMaterial({
        color: color.clone().lerp(new THREE.Color(0xffffff), 0.5),
        transparent: true, opacity: 0.7, depthWrite: false, side: THREE.DoubleSide,
      });
      let cursor = new THREE.Vector3(-1.6, -1.2, 0);
      let heading = new THREE.Vector3(1, 0, 0);
      const pieces = [];

      function newDir(span) {
        const d = new THREE.Vector3(between(-1, 1), between(-0.6, 1), between(-1, 1)).normalize();
        if (d.dot(heading) < -0.3) d.negate();
        return d.multiplyScalar(span);
      }

      const motifs = [];
      for (let m = 0; m < 7; m++) motifs.push(pick(['helix', 'strand', 'loop', 'loop']));
      for (const motif of motifs) {
        if (motif === 'helix') {
          const axis = newDir(1.5);
          const pts = [];
          const turns = between(2.2, 3.4), coilR = 0.24;
          const axN = axis.clone().normalize();
          // 轴接近 (0,1,0) 时叉积会退化，换一个参考向量
          const ref = Math.abs(axN.y) > 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
          const side = ref.cross(axN).normalize();
          const side2 = axN.clone().cross(side).normalize();
          for (let s = 0; s <= 60; s++) {
            const t = s / 60, ang = t * turns * Math.PI * 2;
            pts.push(cursor.clone()
              .addScaledVector(axis, t)
              .addScaledVector(side, Math.cos(ang) * coilR)
              .addScaledVector(side2, Math.sin(ang) * coilR));
          }
          pieces.push({ type: 'helix', curve: new THREE.CatmullRomCurve3(pts) });
          cursor = cursor.clone().add(axis);
          heading.copy(axis).normalize();
        } else if (motif === 'strand') {
          const dirV = newDir(1.1);
          const pts = [cursor.clone(), cursor.clone().add(dirV)];
          pieces.push({ type: 'strand', from: pts[0], to: pts[1] });
          cursor = pts[1];
          heading.copy(dirV).normalize();
        } else {
          const mid1 = cursor.clone().addScaledVector(heading, between(0.3, 0.7))
            .add(new THREE.Vector3(between(-.35, .35), between(.15, .6), between(-.35, .35)));
          const mid2 = cursor.clone().addScaledVector(heading, between(0.8, 1.3))
            .add(new THREE.Vector3(between(-.3, .3), between(-.4, .3), between(-.3, .3)));
          const end = cursor.clone().add(newDir(1.2));
          const curve = new THREE.CatmullRomCurve3([cursor.clone(), mid1, mid2, end]);
          pieces.push({ type: 'loop', curve });
          cursor = end;
          heading.copy(end).sub(mid2).normalize();
        }
      }

      for (const p of pieces) {
        if (p.type === 'helix') {
          group.add(new THREE.Mesh(new THREE.TubeGeometry(p.curve, 120, 0.13, 8, false), mat));
        } else if (p.type === 'loop') {
          // loop 画细一点，否则在屏幕上像一团散线，抢掉螺旋和箭头的主轮廓
          group.add(new THREE.Mesh(new THREE.TubeGeometry(p.curve, 70, 0.038, 6, false), mat));
        } else {
          const w = p.to.clone().sub(p.from);
          const len = w.length();
          const dirN = w.clone().normalize();
          const mid = p.from.clone().addScaledVector(w, 0.5);
          const ribbon = new THREE.Mesh(new THREE.BoxGeometry(len * 0.92, 0.03, len * 0.22), mat);
          ribbon.position.copy(mid);
          // lookAt 对齐的是 +Z，而飘带的长度在 +X，所以直接用四元数把 X 转到链方向
          ribbon.quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), dirN);
          group.add(ribbon);
          const head = new THREE.Mesh(new THREE.ConeGeometry(len * 0.17, len * 0.2, 4), arrowMat);
          head.position.copy(p.to).addScaledVector(dirN, len * 0.06);
          head.lookAt(p.to.clone().add(dirN));
          head.rotateX(Math.PI / 2);
          group.add(head);
        }
      }
      // 活性位点：一小团配体光点
      const lig = new THREE.Points(
        new THREE.BufferGeometry().setFromPoints(shellPoints(12, 0.35, 0)),
        new THREE.PointsMaterial({
          color: 0xfff3b0, size: 0.12, transparent: true, opacity: 0.9,
          blending: THREE.AdditiveBlending, depthWrite: false,
        })
      );
      lig.position.copy(cursor.clone().multiplyScalar(0.4));
      group.add(lig);
      group.scale.setScalar(between(0.85, 1.3));
      return group;
    }
    const proteins = [];
    for (const [hex, pos] of [
      [0x5eead4, [-7.5, -5, -3]], [0xf0abfc, [9.5, 5.5, -6]],
      [0x93c5fd, [-2.5, 7.5, -8]], [0xfda4af, [4, -8.5, -5]],
    ]) {
      const mesh = makeProtein(hex);
      settle(mesh.position.set(...pos), 9);
      scene.add(mesh);
      proteins.push({ mesh, pos, spin: between(0.1, 0.22) * (rand() > 0.5 ? 1 : -1) });
    }

    /* ---------------------------------------------------------- 微生物：杆菌 + 球菌 */
    function helixCurve(len, radius, turns, phase) {
      const pts = [];
      for (let s = 0; s <= 90; s++) {
        const t = s / 90;
        const a = t * turns * Math.PI * 2 + phase;
        pts.push(new THREE.Vector3(t * len, Math.cos(a) * radius * t, Math.sin(a) * radius * t));
      }
      return new THREE.CatmullRomCurve3(pts);
    }
    const microbes = [];
    function makeRod() {
      const group = new THREE.Group();
      const len = between(0.9, 1.5), rad = len * 0.26;
      const mat = new THREE.MeshPhongMaterial({
        color: 0x86efac, emissive: 0x103019, transparent: true, opacity: 0.62,
        shininess: 50, depthWrite: false,
      });
      const body = new THREE.Mesh(new THREE.CapsuleGeometry(rad, len, 8, 18), mat);
      body.rotation.z = Math.PI / 2;
      group.add(body);
      const flagMat = new THREE.MeshBasicMaterial({ color: 0xbbf7d0, transparent: true, opacity: 0.55 });
      const nFlag = 5;
      const xAxis = new THREE.Vector3(1, 0, 0);
      for (let k = 0; k < nFlag; k++) {
        // 周生鞭毛：从菌体表面长出、朝外且偏向后方，而不是贴着菌体平行绕一圈
        const cur = helixCurve(len * 1.5, rad * 1.15, 2.6, k * 1.9);
        const tube = new THREE.Mesh(new THREE.TubeGeometry(cur, 70, rad * 0.05, 5, false), flagMat);
        const theta = rand() * Math.PI * 2;
        const along = between(-0.5, 0.5);
        tube.position.set(along * len, Math.cos(theta) * rad, Math.sin(theta) * rad);
        const outward = new THREE.Vector3(
          -0.5 - Math.abs(along) * 0.5, Math.cos(theta) * 0.95, Math.sin(theta) * 0.95
        ).normalize();
        tube.quaternion.setFromUnitVectors(xAxis, outward);
        group.add(tube);
      }
      // 核区 + 核糖体点
      group.add(new THREE.Points(
        new THREE.BufferGeometry().setFromPoints(shellPoints(16, rad * 0.8, 0)),
        new THREE.PointsMaterial({ color: 0xd9f99d, size: rad * 0.3, transparent: true, opacity: 0.8, blending: THREE.AdditiveBlending, depthWrite: false })
      ));
      return { group, len };
    }
    function makeCocci() {
      const group = new THREE.Group();
      const rad = between(0.28, 0.42);
      const mat = new THREE.MeshPhongMaterial({
        color: 0x99f6e4, emissive: 0x0c3030, transparent: true, opacity: 0.6, depthWrite: false,
      });
      const n = Math.floor(between(4, 8));
      for (let k = 0; k < n; k++) {
        const s = new THREE.Mesh(new THREE.SphereGeometry(rad, 16, 12), mat);
        s.position.set(k * rad * 1.7 - n * rad * 0.85, Math.sin(k * 1.3) * rad * 0.5, Math.cos(k * 0.9) * rad * 0.4);
        group.add(s);
      }
      return { group, len: rad };
    }
    for (let i = 0; i < 6; i++) {
      const built = i % 3 === 2 ? makeCocci() : makeRod();
      const g = built.group;
      g.position.copy(settle(new THREE.Vector3(between(-18, 18), between(-10, 10), between(-10, 2)), 8));
      g.rotation.set(between(-1, 1), between(-1, 1), between(-1, 1));
      const halo = new THREE.Sprite(new THREE.SpriteMaterial({
        map: glowMap, color: 0x86efac, transparent: true, opacity: 0.16,
        blending: THREE.AdditiveBlending, depthWrite: false,
      }));
      halo.scale.setScalar(built.len * 4);
      g.add(halo);
      scene.add(g);
      microbes.push({ group: g, halo, phase: rand() * Math.PI * 2, wig: between(0.6, 1.2) });
    }

    /* ---------------------------------------------------------- 绿藻（衣藻） */
    // 认得出的三件套：杯状叶绿体贴着细胞一侧、红色眼点在杯口、前端两根鞭毛脉冲式划水。
    const algae = [];
    function makeAlga() {
      const group = new THREE.Group();
      const r = between(0.42, 0.75);
      const wall = new THREE.Mesh(
        blobby(r, 0.05, 0.03, 2).scale(1, 1.25, 1),
        new THREE.MeshPhongMaterial({
          color: 0x4ade80, emissive: 0x0d2b16, transparent: true, opacity: 0.4,
          shininess: 90, depthWrite: false,
        })
      );
      group.add(wall);
      const chloro = new THREE.Mesh(
        new THREE.SphereGeometry(r * 0.92, 24, 18, 0, Math.PI * 1.45, Math.PI * 0.16, Math.PI * 0.7),
        new THREE.MeshPhongMaterial({
          color: 0x16a34a, emissive: 0x0b3d1c, transparent: true, opacity: 0.92,
          side: THREE.DoubleSide, depthWrite: false,
        })
      );
      chloro.scale.set(1, 1.3, 1);
      chloro.rotation.y = Math.PI * 0.28;
      group.add(chloro);
      // 蛋白核（淀粉鞘）：叶绿体里的一颗亮点
      const pyrenoid = new THREE.Mesh(
        new THREE.SphereGeometry(r * 0.2, 12, 10),
        new THREE.MeshBasicMaterial({ color: 0xdcfce7, transparent: true, opacity: 0.85 })
      );
      pyrenoid.position.set(r * 0.1, -r * 0.25, r * 0.5);
      group.add(pyrenoid);
      // 眼点：感光的小红斑
      const eyespot = new THREE.Mesh(
        new THREE.SphereGeometry(r * 0.13, 12, 10),
        new THREE.MeshBasicMaterial({ color: 0xff5a3c, transparent: true, opacity: 0.95 })
      );
      eyespot.position.set(r * 0.55, r * 0.35, r * 0.62);
      group.add(eyespot);
      // 两根前鞭毛
      const flagMat = new THREE.MeshBasicMaterial({ color: 0xa7f3d0, transparent: true, opacity: 0.6 });
      const flags = [];
      for (const sgn of [-1, 1]) {
        const cur = new THREE.CatmullRomCurve3([
          new THREE.Vector3(r * 0.9, r * 0.5 * sgn * 0.3, 0),
          new THREE.Vector3(r * 1.7, r * 0.4 * sgn, 0),
          new THREE.Vector3(r * 2.6, r * 0.1 * sgn, r * 0.2 * sgn),
          new THREE.Vector3(r * 3.4, r * 0.6 * sgn, -r * 0.2 * sgn),
        ]);
        const tube = new THREE.Mesh(new THREE.TubeGeometry(cur, 40, r * 0.035, 5, false), flagMat);
        group.add(tube);
        flags.push(tube);
      }
      const halo = new THREE.Sprite(new THREE.SpriteMaterial({
        map: glowMap, color: 0x34d399, transparent: true, opacity: 0.22,
        blending: THREE.AdditiveBlending, depthWrite: false,
      }));
      halo.scale.setScalar(r * 7);
      group.add(halo);
      return { group, chloro, wall, flags, halo, r };
    }
    for (let i = 0; i < 7; i++) {
      const a = makeAlga();
      a.base = settle(new THREE.Vector3(between(-20, 20), between(-9, 8), between(-9, 2)), 10);
      a.group.position.copy(a.base);
      scene.add(a.group);
      algae.push(Object.assign(a, {
        phase: rand() * Math.PI * 2,
        stroke: between(2.2, 3.4),
        cruise: between(0.5, 0.9),
        wob: between(0.3, 0.8),
        target: new THREE.Vector3().copy(a.base),
      }));
    }
    // 四星藻团：4 个细胞围成一个平面星形，远看就是一枚绿色徽记
    (function makeStarColony() {
      const colony = new THREE.Group();
      for (let k = 0; k < 4; k++) {
        const a = makeAlga();
        a.group.scale.setScalar(0.8);
        const ang = (k / 4) * Math.PI * 2;
        a.group.position.set(Math.cos(ang) * 0.9, Math.sin(ang) * 0.9, 0);
        a.group.rotation.z = ang + Math.PI;
        colony.add(a.group);
      }
      settle(colony.position.set(-3.5, -7.5, -2.5), 8);
      scene.add(colony);
      colony.userData.spin = true;
      microbes.push({ group: colony, halo: null, phase: 0, wig: 0.4, isColony: true });
    })();

    /* ---------------------------------------------------------- 萤火虫 */
    // 发光器在腹部末两节，闪烁是"渐亮—急灭"的不对称脉冲，不是正弦；趋光靠近指针。
    const fireflies = [];
    function makeFirefly() {
      const group = new THREE.Group();
      const chitin = new THREE.MeshPhongMaterial({
        color: 0x2a3320, emissive: 0x0b1008, shininess: 60, transparent: true, opacity: 0.95,
      });
      const head = new THREE.Mesh(new THREE.SphereGeometry(0.13, 12, 10), chitin);
      head.position.x = -0.5;
      group.add(head);
      const pronotum = new THREE.Mesh(
        new THREE.SphereGeometry(0.19, 14, 12, 0, Math.PI * 2, 0, Math.PI * 0.55),
        new THREE.MeshPhongMaterial({ color: 0xf5c06a, emissive: 0x3a2408, transparent: true, opacity: 0.9 })
      );
      pronotum.position.set(-0.34, 0.04, 0);
      group.add(pronotum);
      const thorax = new THREE.Mesh(new THREE.SphereGeometry(0.15, 12, 10), chitin);
      thorax.position.set(-0.22, -0.02, 0);
      thorax.scale.set(1.1, 0.85, 0.9);
      group.add(thorax);
      const abd = [];
      for (let k = 0; k < 4; k++) {
        const rad = 0.15 - k * 0.018;
        const lantern = k >= 2;
        const seg = new THREE.Mesh(
          new THREE.SphereGeometry(rad, 12, 10),
          lantern
            ? new THREE.MeshBasicMaterial({ color: 0xffe58a, transparent: true, opacity: 0.95 })
            : chitin
        );
        seg.position.set(0.02 + k * 0.17, -0.01 * k, 0);
        seg.scale.set(1.15, 0.85, 0.9);
        group.add(seg);
        if (lantern) abd.push(seg);
      }
      const wingMat = new THREE.MeshBasicMaterial({
        color: 0xdff0ff, transparent: true, opacity: 0.3, side: THREE.DoubleSide, depthWrite: false,
      });
      const wings = [];
      for (const sgn of [-1, 1]) {
        const w = new THREE.Mesh(new THREE.CircleGeometry(0.34, 14), wingMat);
        w.scale.set(1.5, 0.55, 1);
        w.position.set(-0.05, 0.06, sgn * 0.12);
        w.rotation.set(Math.PI / 2, sgn * 0.25, 0);
        group.add(w);
        wings.push({ mesh: w, sgn });
      }
      const glow = new THREE.Sprite(new THREE.SpriteMaterial({
        map: haloMap, color: 0xffd54a, transparent: true, opacity: 0.5,
        blending: THREE.AdditiveBlending, depthWrite: false,
      }));
      glow.scale.setScalar(1.6);
      glow.position.x = 0.35;
      group.add(glow);
      return { group, abd, wings, glow };
    }
    for (let i = 0; i < 7; i++) {
      const f = makeFirefly();
      scene.add(f.group);
      f.base = new THREE.Vector3(between(-21, 21), between(-9, 8), between(-3, 7));
      // 放大并飞到靠前层，否则在屏幕上只是一粒光点，看不出是昆虫
      f.group.scale.setScalar(between(1.5, 2.3));
      f.phase = rand() * Math.PI * 2;
      f.rate = between(0.55, 1.3);          // 个体闪烁节律不同步，才像真的
      f.amp = between(0.7, 1.4);
      f.sw = between(0.4, 0.9);
      f.group.position.copy(f.base);
      fireflies.push(f);
    }

    /* ---------------------------------------------------------- 远景星尘 */
    const nStars = 900;
    const starPos = new Float32Array(nStars * 3);
    for (let i = 0; i < nStars; i++) {
      starPos[i * 3] = between(-60, 60);
      starPos[i * 3 + 1] = between(-35, 35);
      starPos[i * 3 + 2] = between(-60, -8);
    }
    const starGeo = new THREE.BufferGeometry();
    starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
    const stars = new THREE.Points(starGeo, new THREE.PointsMaterial({
      color: 0x9fc4ff, size: 0.09, transparent: true, opacity: 0.55,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    scene.add(stars);

    /* --------------------------------------------------- 荧光尘埃 / 孢子 / 光斑 */
    // 补深蓝背景的第三层：像免疫荧光染色里的阳性信号点。纯装饰，不读数据、不算数字。
    // 双层反向漂移拉开视差；孢子缓缓上升；焦外光斑给"镜头感"。掉帧时先关光斑和晶格。
    const fluoPal = [0x39e0ff, 0x4affc4, 0xa88bff, 0xff7ad1, 0x7dd3fc];
    const tmpC = new THREE.Color();

    function dustLayer(n, spanY, size, zMin, zMax, opacity) {
      const pos = new Float32Array(n * 3);
      const col = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        pos[i * 3] = between(-70, 70);
        pos[i * 3 + 1] = between(-spanY, spanY);
        pos[i * 3 + 2] = between(zMin, zMax);
        tmpC.setHex(pick(fluoPal));
        const k = 0.55 + rand() * 0.45;          // 同色也分浓淡，才不像贴纸
        col[i * 3] = tmpC.r * k; col[i * 3 + 1] = tmpC.g * k; col[i * 3 + 2] = tmpC.b * k;
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
      const pts = new THREE.Points(geo, new THREE.PointsMaterial({
        size, vertexColors: true, map: haloMap, transparent: true, opacity,
        blending: THREE.AdditiveBlending, depthWrite: false,
      }));
      scene.add(pts);
      return pts;
    }
    // 数量按"软件渲染也能截图"的上限取舍：加得越多越像星空，但大尺寸加色混合的填充率
    // 在没独显的机器上是每帧全屏幕像素级开销，这里宁少勿多（降级阶梯还会再砍一刀）。
    const dustFar = dustLayer(620, 44, 0.34, -66, -16, 0.5);
    const dustNear = dustLayer(260, 28, 0.2, -12, 6, 0.34);

    // 缓缓上浮的荧光孢子：逐细胞更新一次位置数组，代价远小于多一层粒子系统
    const nSpores = 180;
    const sporePos = new Float32Array(nSpores * 3);
    const sporeRise = new Float32Array(nSpores);
    for (let i = 0; i < nSpores; i++) {
      sporePos[i * 3] = between(-44, 44);
      sporePos[i * 3 + 1] = between(-28, 28);
      sporePos[i * 3 + 2] = between(-30, 4);
      sporeRise[i] = between(0.5, 2.1);
    }
    const sporeGeo = new THREE.BufferGeometry();
    sporeGeo.setAttribute('position', new THREE.BufferAttribute(sporePos, 3));
    const spores = new THREE.Points(sporeGeo, new THREE.PointsMaterial({
      color: 0x7cf0d8, size: 0.13, transparent: true, opacity: 0.45,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    scene.add(spores);

    // 焦外光斑：大、极淡（0.035–0.075），只给景深，不给画面加亮度负担。
    // 9 只、半径 3.5–8：加色混合的 sprite 是按像素算钱的，这里刻意压数量。
    const bokeh = [];
    for (let i = 0; i < 9; i++) {
      const b = new THREE.Sprite(new THREE.SpriteMaterial({
        map: glowMap, color: pick(fluoPal), transparent: true,
        opacity: between(0.035, 0.075), blending: THREE.AdditiveBlending, depthWrite: false,
      }));
      b.scale.setScalar(between(3.5, 8));
      b.position.set(between(-48, 48), between(-28, 28), between(-44, -18));
      b.userData = {
        base: b.position.clone(),
        phase: rand() * Math.PI * 2,
        rate: between(0.05, 0.16),
        home: b.material.opacity,
      };
      scene.add(b);
      bokeh.push(b);
    }

    // 极淡的远处晶格：给"太空落落"一个结构骨架，opacity 0.05 不会和细胞抢注意力
    const lattice = new THREE.LineSegments(
      new THREE.WireframeGeometry(new THREE.IcosahedronGeometry(46, 1)),
      new THREE.LineBasicMaterial({
        color: 0x60a5fa, transparent: true, opacity: 0.055,
        blending: THREE.AdditiveBlending, depthWrite: false,
      })
    );
    lattice.position.set(0, 0, -36);
    scene.add(lattice);

    /* ---------------------------------------------------------- 交互 */
    const mouse = { x: 0, y: 0, lastX: null, lastY: null };
    const mouseWorld = new THREE.Vector3(999, 999, 0);
    let excite = 0;

    function onMove(clientX, clientY) {
      const nx = (clientX / window.innerWidth) * 2 - 1;
      const ny = -(clientY / window.innerHeight) * 2 + 1;
      if (mouse.lastX !== null) {
        const speed = Math.hypot(clientX - mouse.lastX, clientY - mouse.lastY);
        excite = Math.min(1, excite + speed * 0.004);
      }
      mouse.lastX = clientX; mouse.lastY = clientY;
      mouse.x = nx; mouse.y = ny;
      const v = new THREE.Vector3(nx, ny, 0.5).unproject(camera);
      const dir = v.sub(camera.position).normalize();
      const t = -camera.position.z / dir.z;
      if (Number.isFinite(t)) mouseWorld.copy(camera.position).addScaledVector(dir, t);
    }
    window.addEventListener('pointermove', (e) => onMove(e.clientX, e.clientY), { passive: true });

    function resize() {
      const w = window.innerWidth, h = window.innerHeight;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    window.addEventListener('resize', resize);
    resize();

    /* ---------------------------------------------------------- 动画 */
    const clock = new THREE.Clock();
    const tmpColor = new THREE.Color();
    const tmpV = new THREE.Vector3();
    const prev = new THREE.Vector3();
    let rafId = null;
    // 掉帧时逐级降载：先降采样，再关掉装饰用的辉光 sprite，宁可朴素也不要卡。
    let avgMs = 16, degraded = 0;

    function frame() {
      const dt = Math.min(clock.getDelta(), 0.05);
      const t = clock.elapsedTime;
      excite *= 0.95;
      const hueShift = 0.06 * Math.sin(t * 0.045);

      avgMs += (dt * 1000 - avgMs) * 0.04;
      if (degraded === 0 && avgMs > 30) {
        degraded = 1;
        renderer.setPixelRatio(1);
        resize();
      } else if (degraded === 1 && avgMs > 46) {
        degraded = 2;
        for (const c of cells) c.halo.visible = false;
        for (const a of algae) a.halo.visible = false;
        for (const m of microbes) if (m.halo) m.halo.visible = false;
        // 光斑和晶格是最先该牺牲的"气氛组"：留着细胞、DNA、蛋白质这些主角
        for (const b of bokeh) b.visible = false;
        lattice.visible = false;
        dustNear.visible = false;
      }

      camera.position.x += (mouse.x * 2.6 - camera.position.x) * 0.035;
      camera.position.y += (mouse.y * 1.8 - camera.position.y) * 0.035;
      camera.lookAt(0, 0, 0);

      for (const c of cells) {
        const { group, base } = c;
        const away = tmpV.copy(group.position).sub(mouseWorld);
        away.z = 0;
        const dist = away.length();
        const target = new THREE.Vector3();
        if (dist < 7 && dist > 0.001) {
          target.copy(away.normalize()).multiplyScalar((1 - dist / 7) * 2.2);
        }
        c.push.lerp(target, 0.06);

        group.position.set(
          base.x + Math.sin(t * c.drift * 0.6 + c.phase) * 0.9 + c.push.x,
          base.y + Math.sin(t * c.drift + c.phase) * 1.1 + c.push.y,
          base.z + Math.cos(t * c.drift * 0.4 + c.phase) * 0.6
        );
        group.rotation.y += c.spin * dt;
        group.rotation.x = Math.sin(t * 0.1 + c.phase) * 0.15;

        tmpColor.setHSL((c.hueBase + hueShift + 1) % 1, 0.82, 0.6 + excite * 0.12);
        c.membrane.material.color.copy(tmpColor);
        c.membrane.material.emissive.copy(tmpColor).multiplyScalar(0.28);
        c.membrane.material.opacity = 0.15 + 0.05 * Math.sin(t * 0.7 + c.phase) + excite * 0.1;
        const nucColor = tmpColor.clone().lerp(new THREE.Color(0xffffff), 0.35);
        c.envelope.material.color.copy(nucColor);
        c.envelope.material.emissive.copy(tmpColor).multiplyScalar(0.5);
        c.nucleus.rotation.y += dt * 0.12;
        c.vesicles.material.color.copy(tmpColor);
        c.vesicles.material.opacity = 0.7 + 0.2 * Math.sin(t * 1.3 + c.phase) + excite * 0.25;
        c.golgi.rotation.y += dt * 0.2;
        for (let k = 0; k < c.mitos.length; k++) {
          c.mitos[k].rotation.z += dt * (0.3 + k * 0.08) * (k % 2 ? 1 : -1);
        }
        c.halo.material.color.copy(tmpColor);
        c.halo.material.opacity = 0.26 + 0.1 * Math.sin(t * 0.5 + c.phase) + excite * 0.28;
      }

      dna.rotation.y = t * 0.14;
      dna2.rotation.y = -t * 0.11;

      for (let i = 0; i < proteins.length; i++) {
        const p = proteins[i];
        p.mesh.rotation.y = t * p.spin;
        p.mesh.rotation.x = Math.sin(t * 0.12 + i) * 0.4;
        p.mesh.position.y = p.pos[1] + Math.sin(t * 0.35 + i * 2) * 0.5;
      }

      for (const m of microbes) {
        if (m.isColony) {
          m.group.rotation.z = t * 0.12;
          continue;
        }
        m.group.rotation.z += dt * 0.25 * m.wig;
        m.group.position.x += Math.sin(t * 0.3 + m.phase) * dt * 0.5;
        m.group.position.y += Math.cos(t * 0.24 + m.phase) * dt * 0.4;
        if (m.halo) m.halo.material.opacity = 0.14 + 0.1 * Math.sin(t * 1.1 + m.phase) + excite * 0.12;
      }

      // 绿藻：鞭毛一收一放的脉冲推进，身体沿前进方向翻滚，被指针轻微吸引
      for (const a of algae) {
        const pull = tmpV.copy(mouseWorld).sub(a.base);
        const near = pull.length();
        const chase = near < 12 ? (1 - near / 12) : 0;
        a.target.copy(a.base).addScaledVector(pull.normalize(), chase * 3.2);
        const stroke = Math.pow(Math.max(0, Math.sin(t * a.stroke + a.phase)), 3);
        a.group.position.lerp(a.target, dt * (0.15 + stroke * 0.5) * a.cruise);
        a.group.position.y += Math.sin(t * a.wob + a.phase) * dt * 0.6;
        a.group.rotation.y = t * 0.3 * a.cruise;
        a.group.rotation.z = Math.sin(t * a.stroke + a.phase) * 0.35;
        const glowUp = 0.18 + stroke * 0.16 + excite * 0.2;
        a.halo.material.opacity = glowUp;
        a.wall.material.color.setHSL((0.36 + hueShift + 1) % 1, 0.72, 0.55 + stroke * 0.12);
        a.chloro.material.emissive.setHex(0x0b3d1c).multiplyScalar(1 + stroke * 0.6);
        for (let k = 0; k < a.flags.length; k++) {
          a.flags[k].rotation.x = Math.sin(t * a.stroke * 2 + a.phase + k * Math.PI) * 0.5;
        }
      }

      // 萤火虫：趋光——指针附近聚拢、加速闪烁；离得远就按自己的节律明灭
      let brightest = -1, brightVal = 0;
      for (let i = 0; i < fireflies.length; i++) {
        const f = fireflies[i];
        const pull = tmpV.copy(mouseWorld).sub(f.base);
        const near = pull.length();
        const chase = near < 14 ? (1 - near / 14) : 0;
        const home = new THREE.Vector3(
          f.base.x + Math.sin(t * 0.21 * f.amp + f.phase) * 3.4 * f.sw,
          f.base.y + Math.sin(t * 0.17 * f.amp + f.phase * 1.7) * 2.2 * f.sw,
          f.base.z + Math.cos(t * 0.13 * f.amp + f.phase) * 1.6
        );
        home.addScaledVector(pull.normalize(), chase * 5.5);
        prev.copy(f.group.position);
        f.group.position.lerp(home, dt * (0.6 + chase * 1.6));
        const vel = tmpV.copy(f.group.position).sub(prev);
        if (vel.lengthSq() > 1e-6) {
          f.group.rotation.y += (Math.atan2(-vel.z, vel.x) - f.group.rotation.y) * 0.08;
          f.group.rotation.z = Math.sin(t * 6 * f.sw + f.phase) * 0.12;
        }
        // 不对称脉冲：慢升快落
        const cyc = ((t * f.rate * (1 + chase * 0.8) + f.phase) % 1 + 1) % 1;
        const blink = Math.pow(Math.max(0, Math.sin(cyc * Math.PI)), 2.4);
        for (const seg of f.abd) {
          seg.material.color.setRGB(1, 0.82 + blink * 0.16, 0.34 + blink * 0.4);
          seg.material.opacity = 0.35 + blink * 0.65;
        }
        f.glow.material.opacity = 0.1 + blink * (0.45 + chase * 0.35) + excite * 0.12;
        f.glow.scale.setScalar(1.1 + blink * (1.8 + chase * 1.4));
        for (const w of f.wings) {
          w.mesh.rotation.y = w.sgn * (0.2 + Math.sin(t * 26 * f.sw + f.phase) * 0.35);
        }
        if (blink > brightVal) { brightVal = blink; brightest = i; }
      }
      if (brightest >= 0) {
        const f = fireflies[brightest];
        lanternLight.position.copy(f.group.position);
        lanternLight.intensity = brightVal * (1.2 + excite);
      }

      stars.rotation.z = t * 0.004;
      stars.material.opacity = 0.45 + 0.12 * Math.sin(t * 0.8);

      // 荧光层：双层反向漂移 + 明暗呼吸，鼠标动一下会整体提亮
      dustFar.rotation.z = t * 0.006;
      dustNear.rotation.z = -t * 0.0045;
      dustFar.position.x = Math.sin(t * 0.05) * 2.2;
      dustNear.position.y = Math.sin(t * 0.07) * 1.5;
      dustFar.material.opacity = 0.46 + 0.1 * Math.sin(t * 0.55) + excite * 0.16;
      dustNear.material.opacity = 0.3 + 0.08 * Math.sin(t * 0.9 + 1.2) + excite * 0.12;
      spores.material.opacity = 0.4 + 0.1 * Math.sin(t * 0.7 + 0.5) + excite * 0.1;
      for (const b of bokeh) {
        const u = b.userData;
        b.position.x = u.base.x + Math.cos(t * u.rate * 0.8 + u.phase) * 2.4;
        b.position.y = u.base.y + Math.sin(t * u.rate + u.phase) * 2.0;
        b.material.opacity = u.home * (0.65 + 0.35 * Math.sin(t * u.rate * 2 + u.phase)) + excite * 0.02;
      }
      lattice.rotation.y = t * 0.012;
      lattice.rotation.x = Math.sin(t * 0.02) * 0.12;
      if (degraded < 2) {
        for (let i = 0; i < nSpores; i++) {
          let y = sporePos[i * 3 + 1] + sporeRise[i] * dt;
          if (y > 28) y = -28;                    // 出顶回底，位置数组原地复用
          sporePos[i * 3 + 1] = y;
          sporePos[i * 3] += Math.sin(t * 0.5 + i) * dt * 0.35;
        }
        sporeGeo.attributes.position.needsUpdate = true;
      }
      keyLight.intensity = 1.35 + excite * 1.2;

      renderer.render(scene, camera);
      rafId = requestAnimationFrame(frame);
    }

    if (reduceMotion) {
      camera.position.set(0, 0, 26);
      renderer.render(scene, camera);
      return;
    }
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        if (rafId !== null) cancelAnimationFrame(rafId);
        rafId = null;
      } else if (rafId === null) {
        clock.getDelta();
        frame();
      }
    });
    frame();
  }

  try { init(); } catch (err) {
    const canvas = document.getElementById('cell-bg');
    if (canvas) canvas.remove();
  }
})();
