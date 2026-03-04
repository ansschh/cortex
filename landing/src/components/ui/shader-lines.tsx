"use client";

import { useEffect, useRef } from "react";

declare global {
  interface Window {
    THREE: any;
  }
}

interface ShaderAnimationProps {
  /** Speed multiplier for the animation */
  speed?: number;
  /** Line glow intensity multiplier */
  intensity?: number;
  /** Custom class name */
  className?: string;
}

export function ShaderAnimation({
  speed = 0.05,
  intensity = 1.0,
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
    };

    const vertexShader = `void main() { gl_Position = vec4(position, 1.0); }`;

    const fragmentShader = `
      precision highp float;
      uniform vec2 resolution;
      uniform float time;
      uniform float intensity;

      float random(in float x) {
        return fract(sin(x) * 1e4);
      }

      void main(void) {
        vec2 uv = (gl_FragCoord.xy * 2.0 - resolution.xy) / min(resolution.x, resolution.y);

        // Circular fade — sphere mask
        float dist = length(uv);
        float sphere = smoothstep(1.0, 0.55, dist);
        if (sphere < 0.001) {
          gl_FragColor = vec4(0.012, 0.027, 0.071, 1.0);
          return;
        }

        vec2 screenSize = vec2(256.0, 256.0);
        vec2 mosaicScale = vec2(4.0, 2.0);
        vec2 uvQ = uv;
        uvQ.x = floor(uvQ.x * screenSize.x / mosaicScale.x) / (screenSize.x / mosaicScale.x);
        uvQ.y = floor(uvQ.y * screenSize.y / mosaicScale.y) / (screenSize.y / mosaicScale.y);

        float t = time * 0.04 + random(uvQ.x) * 0.4;
        float lineWidth = 0.0006 * intensity;

        vec3 color = vec3(0.0);
        for (int j = 0; j < 3; j++) {
          for (int i = 0; i < 5; i++) {
            float fi = float(i);
            float fj = float(j);
            color[j] += lineWidth * fi * fi / abs(fract(t - 0.012 * fj + fi * 0.01) * 1.0 - length(uv));
          }
        }

        // Remap to light blue / white / cyan
        vec3 aurora;
        aurora.r = color[1] * 0.45 + color[2] * 0.2;
        aurora.g = color[1] * 0.6 + color[2] * 0.45;
        aurora.b = color[2] * 1.0 + color[1] * 0.35;

        // Core glow
        float coreGlow = exp(-dist * 3.2) * 0.1 * intensity;
        aurora += vec3(coreGlow * 0.75, coreGlow * 0.88, coreGlow);

        aurora *= sphere;

        vec3 bg = vec3(0.012, 0.027, 0.071);
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
      renderer.setSize(rect.width, rect.height);
      uniforms.resolution.value.x = renderer.domElement.width;
      uniforms.resolution.value.y = renderer.domElement.height;
    };
    onResize();
    window.addEventListener("resize", onResize);

    const animate = () => {
      sceneRef.current.animationId = requestAnimationFrame(animate);
      uniforms.time.value += speed;
      renderer.render(scene, camera);
    };
    animate();
  };

  /** Expose intensity setter for parent to drive reactivity */
  useEffect(() => {
    if (sceneRef.current.uniforms) {
      sceneRef.current.uniforms.intensity.value = intensity;
    }
  }, [intensity]);

  return <div ref={containerRef} className={className ?? "w-full h-full absolute"} />;
}
