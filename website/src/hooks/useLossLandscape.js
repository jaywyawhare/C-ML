import { useEffect, useRef } from 'react'
import * as THREE from 'three'

export function useLossLandscape() {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(window.innerWidth, window.innerHeight)

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(
      42, window.innerWidth / window.innerHeight, 0.1, 300
    )
    camera.position.set(0, 15, 20)
    camera.lookAt(0, 0, 0)

    // ── Loss function — smooth with a bit of character ──
    const SIZE = 24
    const SEG = 160

    function lossHeight(x, z) {
      const sx = x * 0.35, sz = z * 0.35
      // Four smooth basins
      const w1 = 2.8 * Math.exp(-(sx * sx + sz * sz) * 0.5)
      const w2 = 2.2 * Math.exp(-((sx - 2.0) * (sx - 2.0) + (sz + 1.2) * (sz + 1.2)) * 0.4)
      const w3 = 1.8 * Math.exp(-((sx + 1.8) * (sx + 1.8) + (sz - 1.5) * (sz - 1.5)) * 0.35)
      const w4 = 1.4 * Math.exp(-((sx + 0.8) * (sx + 0.8) + (sz + 2.0) * (sz + 2.0)) * 0.45)
      // Gentle ridge
      const r1 = 0.9 * Math.exp(-Math.pow(sz - 0.2 * sx * sx + 0.3, 2) * 1.0) * Math.exp(-sx * sx * 0.025)
      // Mild undulation
      const rip = 0.1 * Math.sin(sx * 3.0) * Math.cos(sz * 3.0)
      // Slope
      const slope = (sx * sx + sz * sz) * 0.015
      // Light texture
      const n = 0.08 * Math.sin(sx * 4.5 + 1.3) * Math.cos(sz * 4.0 + 0.7)
        + 0.04 * Math.sin(sx * 7.0 - sz * 5.0)
      return -(w1 + w2 + w3 + w4) + r1 + rip + slope + n + 3.5
    }

    // ── Build surface ──
    const geo = new THREE.PlaneGeometry(SIZE, SIZE, SEG, SEG)
    geo.rotateX(-Math.PI / 2)
    const pos = geo.attributes.position.array
    let minH = Infinity, maxH = -Infinity

    for (let i = 0; i < pos.length; i += 3) {
      const y = lossHeight(pos[i], pos[i + 2])
      pos[i + 1] = y
      if (y < minH) minH = y
      if (y > maxH) maxH = y
    }
    geo.computeVertexNormals()

    // ── Surface shader ──
    const surfaceUniforms = {
      uMinH: { value: minH },
      uMaxH: { value: maxH },
      uTime: { value: 0 },
      uBallPos: { value: new THREE.Vector3() },
    }

    const surfaceMat = new THREE.ShaderMaterial({
      uniforms: surfaceUniforms,
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: true,

      vertexShader: `
        varying vec3 vPos;
        varying vec3 vNormal;
        varying vec3 vWorldPos;
        varying float vHeight;
        uniform float uMinH, uMaxH;

        void main() {
          vPos = position;
          vNormal = normalize(normalMatrix * normal);
          vWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;
          vHeight = clamp((position.y - uMinH) / (uMaxH - uMinH), 0.0, 1.0);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,

      fragmentShader: `
        varying vec3 vPos;
        varying vec3 vNormal;
        varying vec3 vWorldPos;
        varying float vHeight;

        uniform float uTime;
        uniform vec3 uBallPos;

        vec3 heatmap(float t) {
          vec3 a = vec3(0.02, 0.01, 0.01);
          vec3 b = vec3(0.45, 0.04, 0.02);
          vec3 c = vec3(0.85, 0.22, 0.02);
          vec3 d = vec3(0.96, 0.55, 0.05);
          vec3 e = vec3(1.00, 0.90, 0.50);

          if (t < 0.15) return mix(a, b, t / 0.15);
          if (t < 0.35) return mix(b, c, (t - 0.15) / 0.20);
          if (t < 0.60) return mix(c, d, (t - 0.35) / 0.25);
          return mix(d, e, clamp((t - 0.60) / 0.40, 0.0, 1.0));
        }

        void main() {
          vec3 n = normalize(vNormal);
          vec3 baseColor = heatmap(vHeight);

          vec3 lightDir1 = normalize(vec3(0.5, 1.0, 0.3));
          vec3 lightDir2 = normalize(vec3(-0.4, 0.6, -0.5));
          float diff1 = max(dot(n, lightDir1), 0.0);
          float diff2 = max(dot(n, lightDir2), 0.0) * 0.3;
          float ambient = 0.15;

          vec3 viewDir = normalize(cameraPosition - vWorldPos);
          vec3 halfDir = normalize(lightDir1 + viewDir);
          float spec = pow(max(dot(n, halfDir), 0.0), 40.0) * 0.5 * vHeight;

          // Subtle ball shadow on surface
          float ballDxz = length(vPos.xz - uBallPos.xz);
          float shadow = smoothstep(1.5, 0.3, ballDxz) * 0.15;

          // Grid lines
          float gridScale = 0.8;
          vec2 gridUV = vPos.xz * gridScale;
          vec2 gridAbs = abs(fract(gridUV - 0.5) - 0.5);
          float gridLine = 1.0 - smoothstep(0.01, 0.04, min(gridAbs.x, gridAbs.y));
          float gridFade = 1.0 - smoothstep(6.0, 12.0, length(vPos.xz));
          gridLine *= gridFade * 0.15;
          vec3 gridColor = vec3(0.96, 0.62, 0.08);

          // Edge fade
          float edgeDist = max(abs(vPos.x), abs(vPos.z));
          float edgeFade = 1.0 - smoothstep(9.0, 12.0, edgeDist);

          vec3 color = baseColor * (ambient + diff1 * 0.7 + diff2) * (1.0 - shadow)
                     + vec3(1.0, 0.9, 0.7) * spec
                     + gridColor * gridLine;

          float fresnel = pow(1.0 - max(dot(viewDir, n), 0.0), 4.0);
          color += vec3(0.96, 0.55, 0.10) * fresnel * 0.12;

          gl_FragColor = vec4(color, edgeFade * 0.95);
        }
      `,
    })
    const surface = new THREE.Mesh(geo, surfaceMat)
    scene.add(surface)

    // ── Gradient descent path — multiple restarts to keep it moving ──
    const rawPath = []
    // Run several descents from different starting points, stitch them together
    const starts = [
      { x: 4.5, z: 4.0, vx: -0.08, vz: -0.05 },
      { x: -4.0, z: 3.5, vx: 0.06, vz: -0.07 },
      { x: 3.0, z: -4.0, vx: -0.05, vz: 0.06 },
      { x: -3.5, z: -3.0, vx: 0.07, vz: 0.04 },
    ]
    const STEPS_PER = 300

    for (const start of starts) {
      let cx = start.x, cz = start.z
      let pvx = start.vx, pvz = start.vz
      for (let i = 0; i < STEPS_PER; i++) {
        const h = lossHeight(cx, cz)
        rawPath.push(new THREE.Vector3(cx, h, cz))
        const gx = (lossHeight(cx + 0.01, cz) - lossHeight(cx - 0.01, cz)) / 0.02
        const gz = (lossHeight(cx, cz + 0.01) - lossHeight(cx, cz - 0.01)) / 0.02
        pvx = 0.88 * pvx - 0.05 * gx
        pvz = 0.88 * pvz - 0.05 * gz
        cx += pvx
        cz += pvz
        cx = Math.max(-5, Math.min(5, cx))
        cz = Math.max(-5, Math.min(5, cz))
      }
    }

    const curve = new THREE.CatmullRomCurve3(rawPath, true, 'centripetal', 0.5)
    const totalSteps = starts.length * STEPS_PER
    const pathPoints = curve.getPoints(totalSteps * 3)

    // ── Trail ──
    const TRAIL_MAX = 120
    const trailHistory = []

    function buildTrailMesh() {
      if (trailHistory.length < 4) return null
      const trailCurve = new THREE.CatmullRomCurve3(trailHistory, false, 'centripetal', 0.5)
      const tubeGeo = new THREE.TubeGeometry(trailCurve, trailHistory.length * 2, 0.04, 6, false)

      const count = tubeGeo.attributes.position.count
      const alphas = new Float32Array(count)
      for (let i = 0; i < count; i++) alphas[i] = i / count
      tubeGeo.setAttribute('alpha', new THREE.BufferAttribute(alphas, 1))

      const tubeMat = new THREE.ShaderMaterial({
        transparent: true,
        depthWrite: false,
        vertexShader: `
          attribute float alpha;
          varying float vAlpha;
          void main() {
            vAlpha = alpha;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
        `,
        fragmentShader: `
          varying float vAlpha;
          void main() {
            float a = pow(vAlpha, 2.0) * 0.5;
            vec3 col = mix(vec3(0.3, 0.15, 0.05), vec3(0.7, 0.5, 0.3), vAlpha);
            gl_FragColor = vec4(col, a);
          }
        `,
      })
      return new THREE.Mesh(tubeGeo, tubeMat)
    }

    let trailMesh = null

    // ── Ball — chrome metallic sphere ──
    const ballGeo = new THREE.SphereGeometry(0.4, 48, 48)
    const ballMat = new THREE.ShaderMaterial({
      uniforms: { uTime: { value: 0 }, uCamPos: { value: new THREE.Vector3() } },
      depthTest: true,
      depthWrite: true,
      vertexShader: `
        varying vec3 vNorm;
        varying vec3 vWorldPos;
        varying vec3 vReflect;
        void main() {
          vNorm = normalize(normalMatrix * normal);
          vWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;
          vec3 worldNorm = normalize((modelMatrix * vec4(normal, 0.0)).xyz);
          vec3 viewDir = normalize(cameraPosition - vWorldPos);
          vReflect = reflect(-viewDir, worldNorm);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        varying vec3 vNorm;
        varying vec3 vWorldPos;
        varying vec3 vReflect;
        uniform float uTime;
        uniform vec3 uCamPos;

        // Same heatmap as surface — ball reflects landscape
        vec3 heatmap(float t) {
          vec3 a = vec3(0.02, 0.01, 0.01);
          vec3 b = vec3(0.45, 0.04, 0.02);
          vec3 c = vec3(0.85, 0.22, 0.02);
          vec3 d = vec3(0.96, 0.55, 0.05);
          vec3 e = vec3(1.00, 0.90, 0.50);
          if (t < 0.15) return mix(a, b, t / 0.15);
          if (t < 0.35) return mix(b, c, (t - 0.15) / 0.20);
          if (t < 0.60) return mix(c, d, (t - 0.35) / 0.25);
          return mix(d, e, clamp((t - 0.60) / 0.40, 0.0, 1.0));
        }

        void main() {
          vec3 viewDir = normalize(uCamPos - vWorldPos);
          vec3 n = normalize(vNorm);

          // Fake environment reflection — map reflect direction to heatmap
          float envT = clamp(vReflect.y * 0.5 + 0.5, 0.0, 1.0);
          vec3 envColor = heatmap(envT) * 0.6;

          // Chrome base — dark silver tinted by environment
          vec3 chrome = vec3(0.6, 0.6, 0.65);

          // Fresnel — edges reflect more (like real metal)
          float fresnel = pow(1.0 - max(dot(n, viewDir), 0.0), 4.0);

          // Strong dual specular highlights
          vec3 light1 = normalize(vec3(0.5, 1.0, 0.3));
          vec3 light2 = normalize(vec3(-0.3, 0.8, -0.5));
          float spec1 = pow(max(dot(reflect(-light1, n), viewDir), 0.0), 60.0);
          float spec2 = pow(max(dot(reflect(-light2, n), viewDir), 0.0), 40.0);
          vec3 specular = vec3(1.0, 0.95, 0.85) * spec1 * 1.2
                        + vec3(0.8, 0.85, 1.0) * spec2 * 0.4;

          // Compose — metal = chrome tinted by environment + fresnel + specular
          vec3 color = mix(chrome * 0.3, envColor + chrome * 0.2, fresnel * 0.7 + 0.3)
                     + specular;

          // Slight ambient so it's never invisible
          color += vec3(0.08, 0.08, 0.1);

          gl_FragColor = vec4(color, 1.0);
        }
      `,
    })
    const ball = new THREE.Mesh(ballGeo, ballMat)
    ball.renderOrder = 10
    scene.add(ball)

    // ── Ambient particles ──
    const PARTICLE_COUNT = 150
    const pPositions = new Float32Array(PARTICLE_COUNT * 3)
    const pSizes = new Float32Array(PARTICLE_COUNT)
    const pPhases = new Float32Array(PARTICLE_COUNT)

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      pPositions[i * 3] = (Math.random() - 0.5) * 26
      pPositions[i * 3 + 1] = Math.random() * 6 + 1
      pPositions[i * 3 + 2] = (Math.random() - 0.5) * 26
      pSizes[i] = 0.3 + Math.random() * 1.0
      pPhases[i] = Math.random() * Math.PI * 2
    }

    const particleGeo = new THREE.BufferGeometry()
    particleGeo.setAttribute('position', new THREE.BufferAttribute(pPositions, 3))
    particleGeo.setAttribute('size', new THREE.BufferAttribute(pSizes, 1))

    const particleMat = new THREE.ShaderMaterial({
      uniforms: { uTime: { value: 0 } },
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      vertexShader: `
        attribute float size;
        varying float vDist;
        void main() {
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          vDist = -mv.z;
          gl_PointSize = size * (80.0 / vDist);
          gl_Position = projectionMatrix * mv;
        }
      `,
      fragmentShader: `
        uniform float uTime;
        varying float vDist;
        void main() {
          float d = length(gl_PointCoord - 0.5);
          if (d > 0.5) discard;
          float a = smoothstep(0.5, 0.1, d);
          float fog = smoothstep(50.0, 10.0, vDist);
          float flicker = 0.5 + 0.5 * sin(uTime * 2.0 + vDist * 0.5);
          gl_FragColor = vec4(0.96, 0.65, 0.15, a * fog * flicker * 0.08);
        }
      `,
    })
    scene.add(new THREE.Points(particleGeo, particleMat))

    scene.add(new THREE.AmbientLight(0x332200, 0.5))

    // ── Mouse ──
    let mouseX = 0, mouseY = 0
    const onMove = (e) => {
      mouseX = (e.clientX / window.innerWidth - 0.5) * 2
      mouseY = (e.clientY / window.innerHeight - 0.5) * 2
    }
    window.addEventListener('mousemove', onMove)

    const lookTarget = new THREE.Vector3(0, -1, 0)
    const clock = new THREE.Clock()
    let raf
    let frameCount = 0

    function loop() {
      raf = requestAnimationFrame(loop)
      const t = clock.getElapsedTime()
      frameCount++

      surfaceUniforms.uTime.value = t
      ballMat.uniforms.uTime.value = t
      ballMat.uniforms.uCamPos.value.copy(camera.position)
      particleMat.uniforms.uTime.value = t

      // Camera — nearly static with tiny sway
      const sway = t * 0.012
      const tx = Math.sin(sway) * 1.5 + mouseX * 0.8
      const tz = 20 + Math.cos(sway * 0.7) * 1.0 + mouseY * 0.5
      const ty = 14 + Math.sin(t * 0.06) * 0.3
      camera.position.x += (tx - camera.position.x) * 0.008
      camera.position.z += (tz - camera.position.z) * 0.008
      camera.position.y += (ty - camera.position.y) * 0.008

      // Ball along path
      const totalPts = pathPoints.length
      const rawT = (t * 0.02) % 1
      const pathIdx = rawT * totalPts
      const i0 = Math.floor(pathIdx) % totalPts
      const i1 = (i0 + 1) % totalPts
      const frac = pathIdx - Math.floor(pathIdx)
      const bp = pathPoints[i0].clone().lerp(pathPoints[i1], frac)
      bp.y += 0.6 + Math.sin(t * 3.5) * 0.05

      ball.position.copy(bp)

      // LookAt tracks ball
      const idealLook = new THREE.Vector3(bp.x * 0.5, -0.5, bp.z * 0.5)
      lookTarget.lerp(idealLook, 0.02)
      camera.lookAt(lookTarget)

      // Surface ball position for shadow
      surfaceUniforms.uBallPos.value.copy(bp)

      // Particles
      const pp = particleGeo.attributes.position.array
      for (let i = 0; i < PARTICLE_COUNT; i++) {
        pp[i * 3 + 1] += Math.sin(t * 0.5 + pPhases[i]) * 0.001
        pp[i * 3] += Math.cos(t * 0.3 + pPhases[i]) * 0.0008
      }
      particleGeo.attributes.position.needsUpdate = true

      // Trail
      trailHistory.push(bp.clone())
      if (trailHistory.length > TRAIL_MAX) trailHistory.shift()

      if (frameCount % 4 === 0 && trailHistory.length >= 4) {
        if (trailMesh) {
          scene.remove(trailMesh)
          trailMesh.geometry.dispose()
          trailMesh.material.dispose()
        }
        trailMesh = buildTrailMesh()
        if (trailMesh) scene.add(trailMesh)
      }

      renderer.render(scene, camera)
    }
    loop()

    const onResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight
      camera.updateProjectionMatrix()
      renderer.setSize(window.innerWidth, window.innerHeight)
    }
    window.addEventListener('resize', onResize)

    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('resize', onResize)
      if (trailMesh) { trailMesh.geometry.dispose(); trailMesh.material.dispose() }
      geo.dispose(); surfaceMat.dispose()
      ballGeo.dispose(); ballMat.dispose()
      particleGeo.dispose(); particleMat.dispose()
      renderer.dispose()
    }
  }, [])

  return canvasRef
}
