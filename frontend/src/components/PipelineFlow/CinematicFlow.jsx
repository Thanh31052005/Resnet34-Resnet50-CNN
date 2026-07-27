import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

export default function CinematicFlow({ base64Image }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!base64Image) return;

    const img = new Image();
    img.src = `data:image/png;base64,${base64Image}`;
    
    let isMounted = true;
    let animId;
    let particles = [];
    
    const MAX_PARTICLES = 5000;
    const HEIGHT = 700;
    const WIDTH = 1000;
    const STREAM_WIDTH = 120;

    img.onload = () => {
      if (!isMounted) return;
      const ctx = canvasRef.current.getContext('2d', { alpha: false });

      const offCanvas = document.createElement('canvas');
      offCanvas.width = STREAM_WIDTH; offCanvas.height = STREAM_WIDTH;
      const offCtx = offCanvas.getContext('2d');
      offCtx.drawImage(img, 0, 0, STREAM_WIDTH, STREAM_WIDTH);
      const imgData = offCtx.getImageData(0, 0, STREAM_WIDTH, STREAM_WIDTH).data;

      const getImgColor = (px, py) => {
        const x = Math.floor(px); const y = Math.floor(py);
        if(x<0||x>=STREAM_WIDTH||y<0||y>=STREAM_WIDTH) return [100,100,100];
        const idx = (y * STREAM_WIDTH + x) * 4;
        return [imgData[idx], imgData[idx+1], imgData[idx+2]];
      };

      let spawnIndexY = 0;
      const spawnParticles = (count) => {
        for(let i=0; i<count; i++) {
            const imgX = Math.random() * STREAM_WIDTH;
            const imgY = spawnIndexY % STREAM_WIDTH;
            const color = getImgColor(imgX, imgY);
            
            if (particles.length < MAX_PARTICLES) {
                // Spawn for CNN (left)
                particles.push({
                    x: WIDTH*0.3 - STREAM_WIDTH/2 + imgX, y: -10,
                    vx: 0, vy: 1.5 + Math.random(),
                    r: color[0], g: color[1], b: color[2],
                    size: Math.random() * 1.5 + 0.5,
                    type: 'cnn'
                });
                // Spawn for ResNet (right)
                particles.push({
                    x: WIDTH*0.7 - STREAM_WIDTH/2 + imgX, y: -10,
                    vx: 0, vy: 1.5 + Math.random(),
                    r: color[0], g: color[1], b: color[2], origin: color,
                    size: Math.random() * 1.5 + 0.5,
                    type: 'main'
                });
            }
        }
        spawnIndexY += 1.5;
      };

      let time = 0;

      const drawFrame = () => {
        time++;
        if (time % 2 === 0) spawnParticles(40); 

        ctx.globalCompositeOperation = 'source-over';
        ctx.fillStyle = 'rgba(8, 12, 21, 0.25)';
        ctx.fillRect(0, 0, WIDTH, HEIGHT);
        ctx.globalCompositeOperation = 'screen';

        for (let i = particles.length - 1; i >= 0; i--) {
            const p = particles[i];
            p.x += p.vx; p.y += p.vy;

            // CNN Behavior (Vanishing)
            if (p.type === 'cnn') {
                if (p.y > 150) {
                     p.r *= 0.99; p.g *= 0.99; p.b *= 0.99; // fade out
                     p.vx += (Math.random() - 0.5) * 0.5; // chaos
                }
            }
            // ResNet Behavior (Identity Flow & Merge)
            else if (p.type === 'main') {
                if (p.y > 150 && p.y < 400 && Math.random() < 0.05) {
                     // Split to skip connection
                     p.type = 'skip';
                     p.vx = 2; // move right
                } else if (p.y > 150) {
                     p.r *= 0.995; p.g *= 0.995; p.b *= 0.995;
                }
            }
            else if (p.type === 'skip') {
                if (p.x > WIDTH*0.7 + STREAM_WIDTH/2 + 40) p.vx *= 0.9;
                if (p.y > 380) p.vx -= 0.1; // return to main
                if (p.y > 400 && p.x < WIDTH*0.7 + STREAM_WIDTH/2) {
                     p.type = 'merged';
                     p.vx = 0;
                     // Restore color
                     p.r = p.origin[0]; p.g = p.origin[1]; p.b = p.origin[2];
                }
            }

            ctx.fillStyle = `rgba(${p.r}, ${p.g}, ${p.b}, ${p.type === 'skip' ? 0.8 : 0.6})`;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.type === 'merged' ? p.size*1.5 : p.size, 0, Math.PI*2);
            ctx.fill();

            if (p.y > HEIGHT) particles.splice(i, 1);
        }

        ctx.globalCompositeOperation = 'source-over';
        ctx.fillStyle = "#f43f5e"; ctx.font = "bold 16px sans-serif";
        ctx.fillText("CNN TRUYỀN THỐNG (Mất mát/Vanishing)", WIDTH*0.3 - 150, 40);
        ctx.fillStyle = "#38bdf8"; 
        ctx.fillText("RESNET (Skip Connection + F(x)+x)", WIDTH*0.7 - 130, 40);

        if (isMounted) animId = requestAnimationFrame(drawFrame);
      };
      animId = requestAnimationFrame(drawFrame);
    };

    return () => {
      isMounted = false;
      cancelAnimationFrame(animId);
    };
  }, [base64Image]);

  return (
    <motion.div 
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      className="absolute inset-0 z-50 flex items-center justify-center bg-gray-900"
    >
      <canvas ref={canvasRef} width={1000} height={700} className="rounded-xl border border-gray-700 shadow-2xl" />
    </motion.div>
  );
}
