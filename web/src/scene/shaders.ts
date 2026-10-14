import * as THREE from "three";

// ---------------------------------------------------------------------------
// Coronary artery shader
//   aFlow  : normalised distance downstream of the coronary ostium (baked in build_anatomy.py)
//   uRisk  : predicted stenosis probability -> colour, pulse speed and "turbulence"
// Bright boluses travel from the ostium to the apex; in high-risk vessels they slow
// down and flicker, mimicking delayed contrast filling on an angiogram.
// ---------------------------------------------------------------------------
export const vesselVertex = /* glsl */ `
  attribute float aFlow;
  uniform float uBeat;
  varying vec3 vN;
  varying vec3 vV;
  varying float vFlow;
  void main() {
    vFlow = aFlow;
    vec3 p = position + normal * uBeat;
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    vN = normalize(normalMatrix * normal);
    vV = normalize(-mv.xyz);
    gl_Position = projectionMatrix * mv;
  }
`;

export const vesselFragment = /* glsl */ `
  uniform vec3 uColor;
  uniform float uTime;
  uniform float uRisk;
  uniform float uFlowOn;
  uniform float uEmph;   // 1 = selected/hovered, 0 = normal, -1 = dimmed (another vessel focused)
  varying vec3 vN;
  varying vec3 vV;
  varying float vFlow;

  float hash(float x) { return fract(sin(x * 127.1) * 43758.5453); }

  void main() {
    vec3 N = normalize(vN);
    vec3 L1 = normalize(vec3(0.4, 0.8, 0.6));
    vec3 L2 = normalize(vec3(-0.6, -0.2, 0.4));
    float diff = max(dot(N, L1), 0.0) * 0.75 + max(dot(N, L2), 0.0) * 0.25;
    float rim = pow(1.0 - max(dot(N, normalize(vV)), 0.0), 2.2);
    float spec = pow(max(dot(reflect(-L1, N), normalize(vV)), 0.0), 24.0);

    // contrast boluses: speed falls and spacing tightens as risk rises
    float speed = mix(0.55, 0.12, uRisk);
    float phase = vFlow * mix(9.0, 15.0, uRisk) - uTime * speed * 6.0;
    float band = fract(phase);
    float bolus = smoothstep(0.0, 0.08, band) * (1.0 - smoothstep(0.08, 0.42, band));
    float flicker = 1.0 - uRisk * 0.55 * step(0.6, hash(floor(phase) + floor(uTime * 7.0)));
    float pulse = bolus * flicker * uFlowOn;

    vec3 base = uColor * (0.30 + 0.70 * diff) + vec3(spec) * 0.35;
    vec3 col = base + uColor * rim * 0.9 + mix(vec3(1.0), uColor, 0.35) * pulse * (0.9 + 0.8 * uRisk);
    float emph = clamp(uEmph, 0.0, 1.0);
    col += uColor * emph * (0.45 + 0.35 * sin(uTime * 6.0));
    col *= uEmph < 0.0 ? 0.35 : 1.0;
    gl_FragColor = vec4(col, 1.0);
  }
`;

// ---------------------------------------------------------------------------
// Fresnel "x-ray" shader for context anatomy (skin, bones, lungs).
// ---------------------------------------------------------------------------
export const fresnelVertex = /* glsl */ `
  varying vec3 vN;
  varying vec3 vV;
  varying float vY;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vN = normalize(normalMatrix * normal);
    vV = normalize(-mv.xyz);
    vY = position.y;
    gl_Position = projectionMatrix * mv;
  }
`;

export const fresnelFragment = /* glsl */ `
  uniform vec3 uColor;
  uniform float uOpacity;
  uniform float uPower;
  uniform float uBase;
  uniform float uTime;
  uniform float uScan;   // 1 = animated scan line sweeping the torso
  varying vec3 vN;
  varying vec3 vV;
  varying float vY;
  void main() {
    float f = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), uPower);
    float a = (uBase + f) * uOpacity;
    float scan = uScan * smoothstep(0.03, 0.0, abs(vY - (mod(uTime * 0.6, 7.0) - 3.5))) * f * 0.35;
    gl_FragColor = vec4(uColor * (1.0 + scan), a + scan * uOpacity * 0.6);
  }
`;

export function makeVesselMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader: vesselVertex,
    fragmentShader: vesselFragment,
    uniforms: {
      uColor: { value: new THREE.Color("#e9a23b") },
      uTime: { value: 0 },
      uRisk: { value: 0.5 },
      uFlowOn: { value: 1 },
      uEmph: { value: 0 },
      uBeat: { value: 0 },
    },
  });
}

export function makeFresnelMaterial(color: string, opacity: number, power = 2.0, base = 0.0, scan = 0): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader: fresnelVertex,
    fragmentShader: fresnelFragment,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
    uniforms: {
      uColor: { value: new THREE.Color(color) },
      uOpacity: { value: opacity },
      uPower: { value: power },
      uBase: { value: base },
      uTime: { value: 0 },
      uScan: { value: scan },
    },
  });
}

// ---------------------------------------------------------------------------
// Additive halo around each artery (replaces a bloom post-pass: works on any
// WebGL device, no float render targets). Geometry is inflated along normals.
// ---------------------------------------------------------------------------
export const haloVertex = /* glsl */ `
  uniform float uInflate;
  varying vec3 vN;
  varying vec3 vV;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position + normal * uInflate, 1.0);
    vN = normalize(normalMatrix * normal);
    vV = normalize(-mv.xyz);
    gl_Position = projectionMatrix * mv;
  }
`;
export const haloFragment = /* glsl */ `
  uniform vec3 uColor;
  uniform float uStrength;
  varying vec3 vN;
  varying vec3 vV;
  void main() {
    float f = pow(max(dot(normalize(vN), normalize(vV)), 0.0), 1.6);
    gl_FragColor = vec4(uColor * f * uStrength, 1.0);
  }
`;

export function makeHaloMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader: haloVertex,
    fragmentShader: haloFragment,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    side: THREE.BackSide,
    uniforms: {
      uColor: { value: new THREE.Color("#e9a23b") },
      uStrength: { value: 0.35 },
      uInflate: { value: 0.022 },
    },
  });
}
