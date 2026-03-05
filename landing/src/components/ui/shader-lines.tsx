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
  /** When true, removes sphere mask and uses finer lines */
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
    };

    const vertexShader = `void main() { gl_Position = vec4(position, 1.0); }`;

    const fragmentShader = `
      precision highp float;
      uniform vec2 resolution;
      uniform float time;
      uniform float intensity;
      uniform float expanded;

      float random(in float x) {
        return fract(sin(x) * 1e4);
      }

      void main(void) {
        vec2 uv = (gl_FragCoord.xy * 2.0 - resolution.xy) / min(resolution.x, resolution.y);

        float dist = length(uv);

        // Sphere mask: fades out when expanded (ripples reach edges)
        float sphereScale = 1.0 + (intensity - 1.0) * 0.25;
        float sphereDist = dist / sphereScale;
        // When expanded=1, sphere=1 everywhere (no mask). When expanded=0, normal sphere mask.
        float sphereMask = smoothstep(1.0, 0.45, sphereDist);
        float sphere = mix(sphereMask, 1.0, expanded);

        // Edge fade for expanded mode — gentle vignette so it doesn't clip hard
        float edgeFade = mix(1.0, smoothstep(1.8, 0.3, dist), expanded);
        sphere *= edgeFade;

        vec3 bg = vec3(0.012, 0.027, 0.071);

        if (sphere < 0.001) {
          gl_FragColor = vec4(bg, 1.0);
          return;
        }

        vec2 screenSize = vec2(256.0, 256.0);
        vec2 mosaicScale = vec2(4.0, 2.0);
        vec2 uvQ = uv;
        uvQ.x = floor(uvQ.x * screenSize.x / mosaicScale.x) / (screenSize.x / mosaicScale.x);
        uvQ.y = floor(uvQ.y * screenSize.y / mosaicScale.y) / (screenSize.y / mosaicScale.y);

        float t = time * 0.04 + random(uvQ.x) * 0.4;

        // Lines get much finer when expanded (0.0008 → 0.0002)
        float baseWidth = mix(0.0008, 0.0002, expanded);
        float lineWidth = baseWidth * intensity;

        vec3 color = vec3(0.0);
        for (int j = 0; j < 3; j++) {
          for (int i = 0; i < 5; i++) {
            float fi = float(i);
            float fj = float(j);
            color[j] += lineWidth * fi * fi / abs(fract(t - 0.012 * fj + fi * 0.01) * 1.0 - dist);
          }
        }

        // Remap to light blue / white / cyan
        vec3 aurora;
        aurora.r = color[1] * 0.45 + color[2] * 0.2;
        aurora.g = color[1] * 0.6 + color[2] * 0.45;
        aurora.b = color[2] * 1.0 + color[1] * 0.35;

        // Core glow — reduced when expanded
        float glowStrength = mix(0.15, 0.04, expanded);
        float coreGlow = exp(-dist * 2.5) * glowStrength * intensity;
        aurora += vec3(coreGlow * 0.75, coreGlow * 0.88, coreGlow);

        aurora *= sphere;

        gl_FragColor = vec4(bg + aurora, 1.0);
      }
    `;

    const material = new THREE.ShaderMaterial({
      uniforms,
      vertexShader,
      fragmentShader,
    });

    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
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

  // Drive intensity uniform
  useEffect(() => {
    if (sceneRef.current.uniforms) {
      sceneRef.current.uniforms.intensity.value = intensity;
    }
  }, [intensity]);

  // Drive expanded uniform (smooth transition handled by animation loop)
  useEffect(() => {
    const target = expanded ? 1.0 : 0.0;
    if (!sceneRef.current.uniforms) return;

    // Animate the expanded uniform smoothly
    let current = sceneRef.current.uniforms.expanded.value;
    let raf: number;
    const ease = () => {
      current += (target - current) * 0.04; // slow, smooth transition
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
