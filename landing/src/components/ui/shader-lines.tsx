"use client";

import { useEffect, useRef } from "react";

declare global {
  interface Window {
    THREE: any;
  }
}

interface ShaderAnimationProps {
  speed?: number;
  intensity?: number;
  expanded?: boolean;
  className?: string;
}

export function ShaderAnimation({
  speed = 0.05,
  intensity = 1.0,
  expanded = false,
  className,
}: ShaderAnimationProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<{
    camera: any;
    scene: any;
    renderer: any;
    uniforms: any;
    animationId: number | null;
  }>({
    camera: null,
    scene: null,
    renderer: null,
    uniforms: null,
    animationId: null,
  });

  useEffect(() => {
    const script = document.createElement("script");
    script.src =
      "https://cdnjs.cloudflare.com/ajax/libs/three.js/89/three.min.js";
    script.onload = () => {
      if (containerRef.current && window.THREE) {
        initThreeJS();
      }
    };
    document.head.appendChild(script);

    return () => {
      if (sceneRef.current.animationId) {
        cancelAnimationFrame(sceneRef.current.animationId);
      }
      if (sceneRef.current.renderer) {
        sceneRef.current.renderer.dispose();
      }
      if (script.parentNode) {
        document.head.removeChild(script);
      }
    };
  }, []);

  const initThreeJS = () => {
    if (!containerRef.current || !window.THREE) return;

    const THREE = window.THREE;
    const container = containerRef.current;

    container.innerHTML = "";

    const camera = new THREE.Camera();
    camera.position.z = 1;

    const scene = new THREE.Scene();
    const geometry = new THREE.PlaneBufferGeometry(2, 2);

    const uniforms = {
      time: { type: "f", value: 1.0 },
      resolution: { type: "v2", value: new THREE.Vector2() },
      intensity: { type: "f", value: intensity },
      expanded: { type: "f", value: 0.0 },
      pulse: { type: "f", value: 0.0 },
    };

    const vertexShader = `void main() { gl_Position = vec4(position, 1.0); }`;

    const fragmentShader = `
      precision highp float;
      uniform vec2 resolution;
      uniform float time;
      uniform float intensity;
      uniform float expanded;
      uniform float pulse;

      float random(in float x) {
        return fract(sin(x) * 1e4);
      }

      void main(void) {
        vec2 uv = (gl_FragCoord.xy * 2.0 - resolution.xy) / min(resolution.x, resolution.y);

        float dist = length(uv);

        // Sphere mask with smooth alpha edges
        float sphereScale = 1.0 + (intensity - 1.0) * 0.25;
        float sphereDist = dist / sphereScale;
        // Very wide, soft edge (0.3 range) — never a hard cutoff
        float sphereMask = smoothstep(1.05, 0.35, sphereDist);
        float sphere = mix(sphereMask, 1.0, expanded);

        // Expanded mode: gentle vignette
        float edgeFade = mix(1.0, smoothstep(1.8, 0.3, dist), expanded);
        sphere *= edgeFade;

        if (sphere < 0.002) {
          gl_FragColor = vec4(0.0, 0.0, 0.0, 0.0);
          return;
        }

        // Pulse: radially push/pull the distance used for line positions
        // This makes lines breathe in and out with audio
        float pulsedDist = dist + pulse * 0.15 * sin(dist * 6.0 - time * 0.3);

        vec2 screenSize = vec2(256.0, 256.0);
        vec2 mosaicScale = vec2(4.0, 2.0);
        vec2 uvQ = uv;
        uvQ.x = floor(uvQ.x * screenSize.x / mosaicScale.x) / (screenSize.x / mosaicScale.x);
        uvQ.y = floor(uvQ.y * screenSize.y / mosaicScale.y) / (screenSize.y / mosaicScale.y);

        float t = time * 0.04 + random(uvQ.x) * 0.4;

        // Lines finer when expanded
        float baseWidth = mix(0.0008, 0.0002, expanded);
        float lineWidth = baseWidth * intensity;

        vec3 color = vec3(0.0);
        for (int j = 0; j < 3; j++) {
          for (int i = 0; i < 5; i++) {
            float fi = float(i);
            float fj = float(j);
            // Use pulsedDist instead of dist for the line distance calc
            color[j] += lineWidth * fi * fi / abs(fract(t - 0.012 * fj + fi * 0.01) * 1.0 - pulsedDist);
          }
        }

        // Remap to light blue / white / cyan
        vec3 aurora;
        aurora.r = color[1] * 0.45 + color[2] * 0.2;
        aurora.g = color[1] * 0.6 + color[2] * 0.45;
        aurora.b = color[2] * 1.0 + color[1] * 0.35;

        // Core glow
        float glowStrength = mix(0.15, 0.04, expanded);
        float coreGlow = exp(-dist * 2.5) * glowStrength * intensity;
        aurora += vec3(coreGlow * 0.75, coreGlow * 0.88, coreGlow);

        aurora *= sphere;

        // Output with alpha — edges fade to transparent, not to a bg color
        float alpha = sphere;
        gl_FragColor = vec4(aurora, alpha);
      }
    `;

    const material = new THREE.ShaderMaterial({
      uniforms,
      vertexShader,
      fragmentShader,
      transparent: true,
    });

    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setClearColor(0x000000, 0);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    sceneRef.current = { camera, scene, renderer, uniforms, animationId: null };

    const onResize = () => {
      const rect = container.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return;
      renderer.setSize(rect.width, rect.height);
      uniforms.resolution.value.x = renderer.domElement.width;
      uniforms.resolution.value.y = renderer.domElement.height;
    };
    onResize();
    window.addEventListener("resize", onResize);

    const resizeObserver = new ResizeObserver(() => onResize());
    resizeObserver.observe(container);

    const animate = () => {
      sceneRef.current.animationId = requestAnimationFrame(animate);
      uniforms.time.value += speed;
      renderer.render(scene, camera);
    };
    animate();
  };

  // Drive intensity + pulse uniforms together
  useEffect(() => {
    if (sceneRef.current.uniforms) {
      sceneRef.current.uniforms.intensity.value = intensity;
      // pulse is driven by how far intensity is from idle (1.0)
      // This makes lines breathe when audio is playing
      sceneRef.current.uniforms.pulse.value = Math.max(0, intensity - 1.0);
    }
  }, [intensity]);

  // Smooth expanded transition
  useEffect(() => {
    const target = expanded ? 1.0 : 0.0;
    if (!sceneRef.current.uniforms) return;

    let current = sceneRef.current.uniforms.expanded.value;
    let raf: number;
    const ease = () => {
      current += (target - current) * 0.04;
      if (Math.abs(current - target) < 0.005) {
        current = target;
      }
      sceneRef.current.uniforms.expanded.value = current;
      if (current !== target) {
        raf = requestAnimationFrame(ease);
      }
    };
    raf = requestAnimationFrame(ease);
    return () => cancelAnimationFrame(raf);
  }, [expanded]);

  return <div ref={containerRef} className={className ?? "w-full h-full absolute"} />;
}
