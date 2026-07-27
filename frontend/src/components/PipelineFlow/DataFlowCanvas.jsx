import { useEffect, useRef } from 'react';

/**
 * DataFlowCanvas.jsx
 * ==================
 * Render hoạt ảnh mượt mà ở mức pixel (particle system) để mô phỏng sự khác biệt 
 * giữa luồng dữ liệu của CNN Standard (mất mát thông tin) và ResNet (Identity Skip).
 */

const NUM_PARTICLES = 150;

export default function DataFlowCanvas({ width = 300, height = 350, isResnet = false }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d', { alpha: false });
    
    let animationFrameId;

    // --- Cấu trúc Particle ---
    const particles = Array.from({ length: NUM_PARTICLES }).map(() => resetParticle(true));

    function resetParticle(initial = false) {
      return {
        x: width / 2 + (Math.random() - 0.5) * 40,
        y: initial ? Math.random() * height : -10, // Rơi từ trên xuống
        vy: 1.5 + Math.random() * 1.5,
        vx: (Math.random() - 0.5) * 0.5,
        energy: 1.0,  // Độ sáng / sức mạnh của feature
        color: [99, 102, 241], // Màu RGB gốc (indigo)
        isSkip: isResnet && Math.random() > 0.5, // 50% hạt đi qua đường skip (nếu là ResNet)
      };
    }

    const draw = () => {
      // Fade mờ dần để tạo hiệu ứng trail (đuôi sáng)
      ctx.globalCompositeOperation = 'source-over';
      ctx.fillStyle = 'rgba(10, 14, 26, 0.2)'; // Màu nền
      ctx.fillRect(0, 0, width, height);

      // Điểm cộng sáng (glowing blending)
      ctx.globalCompositeOperation = 'lighter';

      const blockYStart = 80;
      const blockYEnd = height - 60;

      for (let p of particles) {
        // Cập nhật vị trí
        p.y += p.vy;
        p.x += p.vx;

        // Vòng lặp
        if (p.y > height + 20) {
          Object.assign(p, resetParticle(false));
        }

        let currentEnergy = p.energy;
        let pX = p.x;

        if (p.y > blockYStart && p.y < blockYEnd) {
          if (p.isSkip) {
            // Hạt đi đường vòng (Skip Connection - hình vòng cung bên trái)
            const progress = (p.y - blockYStart) / (blockYEnd - blockYStart);
            const curve = Math.sin(progress * Math.PI) * 70; // Bán kính vòng
            pX = width / 2 - curve - 20 + p.vx * 10;
            // Energy được giữ nguyên hoàn toàn
            ctx.fillStyle = `rgb(34, 197, 94)`; // Màu xanh lá cho đường skip
          } else {
            // Hạt đi đường trục chính (Main Block) - mất dần energy
            currentEnergy *= 0.98; // Giảm dần năng lượng qua các hidden layers
            p.energy = currentEnergy;
            
            // Xoay màu về phía đỏ bầm/xám khi mất thông tin
            ctx.fillStyle = `rgba(${p.color[0] * currentEnergy}, ${p.color[1] * currentEnergy * 0.5}, ${p.color[2] * currentEnergy}, ${currentEnergy})`;
            
            // Tăng xao nhãng (distortion) bằng cách rung lắc x:
            pX += (Math.random() - 0.5) * 2 * (1 - currentEnergy);
          }
        } else {
          // Khi đã thoát khỏi block về đích (Add/Merge phần Identity lại)
          if (p.y >= blockYEnd && p.y <= blockYEnd + 20) {
              // Phục hồi nhẹ hoặc gộp
              if (isResnet) {
                  // Phục hồi màu về độ chói sáng (màu cyan / trắng)
                  ctx.fillStyle = `rgba(56, 189, 248, 1)`;
              } else {
                  // Standard CNN: Yếu nhợt nhạt tàn lụi
                  ctx.fillStyle = `rgba(${p.color[0] * p.energy}, 50, 50, ${p.energy})`;
              }
          } else {
             // Basic state trước và sau
             const a = isResnet ? 1 : Math.max(0.2, p.energy);
             ctx.fillStyle = `rgba(${p.color[0]}, ${p.color[1]}, ${p.color[2]}, ${a})`;
          }
        }

        // Vẽ hạt ánh sáng
        ctx.beginPath();
        const size = (p.isSkip ? 2.5 : 2) * currentEnergy;
        ctx.arc(pX, p.y, size > 0 ? size : 0.1, 0, Math.PI * 2);
        ctx.fill();

        // Glow
        if (currentEnergy > 0.4 || p.isSkip) {
          ctx.beginPath();
          ctx.arc(pX, p.y, size * 2.5, 0, Math.PI * 2);
          ctx.globalAlpha = 0.3;
          ctx.fill();
          ctx.globalAlpha = 1.0;
        }
      }

      // --- Vẽ Background Decor ---
      ctx.globalCompositeOperation = 'source-over';
      ctx.strokeStyle = 'rgba(255,255,255,0.05)';
      ctx.lineWidth = 1;
      
      // Vẽ Box đại diện cho Convolution Block
      ctx.setLineDash([5, 5]);
      ctx.strokeRect(width / 2 - 35, blockYStart, 70, blockYEnd - blockYStart);
      ctx.fillStyle = 'rgba(255,255,255,0.5)';
      ctx.font = '10px Inter';
      ctx.textAlign = 'center';
      ctx.fillText("H(x)", width / 2, (blockYStart + blockYEnd) / 2);

      // Nếu ResNet, vẽ line chỉ dẫn skip
      if (isResnet) {
        ctx.beginPath();
        ctx.moveTo(width / 2, blockYStart - 10);
        ctx.quadraticCurveTo(width / 2 - 100, (blockYStart + blockYEnd) / 2, width / 2, blockYEnd + 10);
        ctx.strokeStyle = 'rgba(34, 197, 94, 0.4)';
        ctx.setLineDash([]);
        ctx.stroke();

        ctx.fillStyle = 'rgba(34, 197, 94, 0.8)';
        ctx.fillText("Identity (x)", width / 2 - 50, (blockYStart + blockYEnd) / 2);
        
        // Plus symbol
        ctx.beginPath();
        ctx.arc(width/2, blockYEnd + 15, 8, 0, Math.PI*2);
        ctx.fillStyle = '#1e293b';
        ctx.fill();
        ctx.strokeStyle = '#22c55e';
        ctx.stroke();
        ctx.fillStyle = '#22c55e';
        ctx.fillText("+", width/2, blockYEnd + 18);
      }

      animationFrameId = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [width, height, isResnet]);

  return (
    <canvas 
      ref={canvasRef} 
      width={width} 
      height={height} 
      style={{ display: 'block', borderRadius: '8px', border: '1px solid var(--border-color)', background: 'var(--bg-primary)' }}
    />
  );
}
