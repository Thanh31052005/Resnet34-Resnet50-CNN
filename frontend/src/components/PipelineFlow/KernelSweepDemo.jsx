import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './KernelSweepDemo.css';

/**
 * KernelSweepDemo.jsx
 * ==================
 * Trực quan hóa phép quét Conv 7x7 (Kernel Sweep).
 * Mô phỏng cách bộ lọc trích xuất đặc trưng từ ảnh gốc.
 */
export default function KernelSweepDemo({ originalImage, featureMap }) {
  const [pos, setPos] = useState({ x: 0, y: 0 });
  const gridSize = 14; // Lưới ảo để quét (ví dụ 14x14 bước cho demo)
  const [step, setStep] = useState(0);

  // Chạy animation quét tự động
  useEffect(() => {
    const interval = setInterval(() => {
      setStep((prev) => (prev + 1) % (gridSize * gridSize));
    }, 1200);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const x = step % gridSize;
    const y = Math.floor(step / gridSize);
    setPos({ x, y });
  }, [step]);

  // Tỷ lệ phần trăm để di chuyển overlay
  const leftPct = (pos.x / gridSize) * 100;
  const topPct = (pos.y / gridSize) * 100;

  return (
    <div className="kernel-sweep">
      <div className="kernel-sweep__container">
        
        {/* PANEL 1: INPUT IMAGE + SCANNING WINDOW */}
        <div className="kernel-sweep__panel">
          <div className="kernel-sweep__label">ẢNH GỐC (INPUT 224x224)</div>
          <div className="kernel-sweep__viewport">
            <img 
              src={`data:image/png;base64,${originalImage}`} 
              className="kernel-sweep__image" 
              alt="Input" 
            />
            {/* Scanning Box */}
            <motion.div 
               className="kernel-sweep__window"
               animate={{ left: `${leftPct}%`, top: `${topPct}%` }}
               transition={{ type: 'spring', stiffness: 80, damping: 15 }}
            >
               <div className="kernel-sweep__window-grid">
                  {[...Array(9)].map((_, i) => <div key={i} className="grid-line" />)}
               </div>
               {/* Laser Beam effect (conceptual) */}
               <div className="kernel-sweep__beam" />
            </motion.div>
          </div>
          <div className="kernel-sweep__status">
             Kernel: 7×7 | Stride: 2
          </div>
        </div>

        {/* CONNECTING HUB (Magnifier / Dot Product) */}
        <div className="kernel-sweep__hub">
           <div className="kernel-sweep__magnifier">
              <div className="magnifier-box">
                 <div className="magnifier-label">Phóng đại 7×7</div>
                 <div className="magnifier-grid">
                    {[...Array(49)].map((_, i) => (
                       <motion.div 
                          key={i} 
                          className="magnifier-pixel"
                          animate={{ opacity: [0.3, 1, 0.3], scale: [1, 1.1, 1] }}
                          transition={{ duration: 2, repeat: Infinity, delay: i * 0.02 }}
                       />
                    ))}
                 </div>
              </div>
              <div className="kernel-sweep__math-sign">×</div>
              <div className="magnifier-box magnifier-box--weights">
                 <div className="magnifier-label">Weights (Bộ lọc)</div>
                 <div className="magnifier-grid">
                    {[...Array(49)].map((_, i) => (
                       <div key={i} className="magnifier-weight" style={{ opacity: Math.random() }} />
                    ))}
                 </div>
              </div>
           </div>
           <div className="kernel-sweep__result-line">∑ (w · x) + b = y</div>
        </div>

        {/* PANEL 2: FEATURE MAP OUTPUT */}
        <div className="kernel-sweep__panel">
          <div className="kernel-sweep__label">FEATURE MAP (OUTPUT)</div>
          <div className="kernel-sweep__viewport kernel-sweep__viewport--output">
            <img 
              src={`data:image/png;base64,${featureMap}`} 
              className="kernel-sweep__image" 
              alt="Feature Map" 
            />
            {/* Active Output Pixel */}
            <motion.div 
               className="kernel-sweep__pixel-out"
               animate={{ left: `${leftPct}%`, top: `${topPct}%` }}
               transition={{ type: 'spring', stiffness: 80, damping: 15 }}
            />
          </div>
          <div className="kernel-sweep__status">
             Kích hoạt đặc trưng (Bản đồ nhiệt)
          </div>
        </div>

      </div>

      <div className="kernel-sweep__footer">
         <div className="info-tag">💡 Thuật toán Quét: Kernel 7x7 trượt qua ảnh từng pixel để nhận diện các hình thái đơn giản (cạnh, nét cong).</div>
      </div>
    </div>
  );
}
