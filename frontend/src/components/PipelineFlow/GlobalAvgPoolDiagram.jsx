import React, { useMemo, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

/**
 * GlobalAvgPoolDiagram.jsx
 * ========================
 * Scientific Visualization using BEM and Canvas for performance.
 * Layout: Bar Chart | 2048 Canvas Heatmap | Stats
 */

// --- Sub-component: Pooling Mechanism Demo (Real Pixel Data) ---
const PoolingMechanismDemo = ({ originalImage }) => {
  const [activeIdx, setActiveIdx] = React.useState(0);
  const [showNumbers, setShowNumbers] = React.useState(true);
  const [pixelMatrix, setPixelMatrix] = React.useState([]);
  
  const size = 8; // 8x8 view of real pixels
  const windowSize = 2;
  const stride = 2;
  const totalSteps = 16; // (8/2) * (8/2)

  // EXTRACT REAL PIXELS FROM IMAGE OR USE MOCK
  React.useEffect(() => {
    if (!originalImage) {
      // Fallback: Random but interesting grayscale patterns
      const mock = Array.from({ length: size }, (_, r) => 
        Array.from({ length: size }, (_, c) => Math.round(50 + Math.random() * 200))
      );
      setPixelMatrix(mock);
      return;
    }
    
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      canvas.width = size;
      canvas.height = size;
      ctx.drawImage(img, 0, 0, size, size);
      const imageData = ctx.getImageData(0, 0, size, size).data;
      const matrix = [];
      for (let i = 0; i < size; i++) {
        const row = [];
        for (let j = 0; j < size; j++) {
          const idx = (i * size + j) * 4;
          const gray = Math.round(0.299 * imageData[idx] + 0.587 * imageData[idx+1] + 0.114 * imageData[idx+2]);
          row.push(gray);
        }
        matrix.push(row);
      }
      setPixelMatrix(matrix);
    };
    img.src = `data:image/png;base64,${originalImage}`;
  }, [originalImage]);

  React.useEffect(() => {
    const timer = setInterval(() => {
      setActiveIdx((prev) => (prev + 1) % totalSteps);
    }, 1500);
    return () => clearInterval(timer);
  }, [pixelMatrix]);

  if (pixelMatrix.length === 0) return (
    <div className="text-gray-500 italic text-xs p-4">Đang khởi tạo dữ liệu pixel...</div>
  );

  // Calculate window position
  const getWindowPos = (idx) => {
    const stepsPerRow = size / stride;
    const r = Math.floor(idx / stepsPerRow) * stride;
    const c = (idx % stepsPerRow) * stride;
    return { r, c };
  };

  const { r: startR, c: startC } = getWindowPos(activeIdx);
  
  // Get current window values
  const windowValues = [];
  for (let i = 0; i < windowSize; i++) {
    for (let j = 0; j < windowSize; j++) {
      windowValues.push(pixelMatrix[startR + i][startC + j]);
    }
  }

  const maxVal = Math.max(...windowValues);
  const avgVal = Math.round(windowValues.reduce((a, b) => a + b, 0) / windowValues.length);

  return (
    <div className="pooling-demo">
      <div className="pooling-demo__header-alt">
         <div className="pooling-demo__label">DỮ LIỆU PIXEL THỰC TẾ (SAMPLED 8×8)</div>
         <button 
           className="pipeline-node__tag" 
           style={{ cursor: 'pointer', border: '1px solid #38bdf8' }}
           onClick={() => setShowNumbers(!showNumbers)}
         >
           {showNumbers ? 'Ẩn số' : 'Hiện số'}
         </button>
      </div>

      <div className="pooling-demo__grid-container">
        {/* INPUT REAL PIXEL GRID */}
        <div className="pooling-demo__panel">
          <div className="pooling-demo__grid pooling-demo__grid--input8">
            {pixelMatrix.flat().map((val, i) => {
              const r = Math.floor(i / size);
              const c = i % size;
              const isInWindow = (r >= startR && r < startR + windowSize) && (c >= startC && c < startC + windowSize);
              return (
                <div 
                  key={i} 
                  className={`pooling-demo__cell ${isInWindow ? 'pooling-demo__cell--active' : ''}`}
                  style={{ 
                    backgroundColor: `rgb(${val}, ${val}, ${val})`,
                    color: val > 128 ? '#000' : '#fff',
                    fontSize: '7px'
                  }}
                >
                  {showNumbers && val}
                </div>
              );
            })}
            <motion.div 
               className="pooling-demo__window"
               style={{ width: (windowSize/size)*100 + '%', height: (windowSize/size)*100 + '%' }}
               animate={{ top: (startR/size)*100 + '%', left: (startC/size)*100 + '%' }}
               transition={{ type: 'spring', stiffness: 120, damping: 20 }}
            />
          </div>
          <div className="pooling-demo__math">Cửa sổ trượt: {windowSize}×{windowSize} | Bước nhảy: {stride}</div>
        </div>

        <div className="pooling-demo__arrow">→</div>

        <div className="pooling-demo__results">
           <div className="pooling-demo__panel pooling-demo__panel--max">
              <div className="pooling-demo__label">MAX POOLING (CỰC ĐẠI)</div>
              <div className="pooling-demo__grid pooling-demo__grid--output4">
                 {Array.from({ length: totalSteps }).map((_, i) => (
                   <div 
                     key={i} 
                     className={`pooling-demo__cell ${activeIdx === i ? 'pooling-demo__cell--result' : ''}`}
                     style={activeIdx === i ? { backgroundColor: `rgb(${maxVal}, ${maxVal}, ${maxVal})`, color: maxVal > 128 ? '#000' : '#fff' } : {}}
                   >
                      {activeIdx === i ? maxVal : ''}
                   </div>
                 ))}
              </div>
              <div className="pooling-demo__math-op">Pick Max Intensity: <span className="highlight-max">{maxVal}</span></div>
           </div>

           <div className="pooling-demo__panel pooling-demo__panel--avg">
              <div className="pooling-demo__label">AVG POOLING (TRUNG BÌNH)</div>
              <div className="pooling-demo__grid pooling-demo__grid--output4">
                 {Array.from({ length: totalSteps }).map((_, i) => (
                   <div 
                     key={i} 
                     className={`pooling-demo__cell ${activeIdx === i ? 'pooling-demo__cell--result' : ''}`}
                     style={activeIdx === i ? { backgroundColor: `rgb(${avgVal}, ${avgVal}, ${avgVal})`, color: avgVal > 128 ? '#000' : '#fff' } : {}}
                   >
                      {activeIdx === i ? avgVal : ''}
                   </div>
                 ))}
              </div>
              <div className="pooling-demo__math-op">Average Blending: <span className="highlight-avg">{avgVal}</span></div>
           </div>
        </div>
      </div>
    </div>
  );
};

export default function GlobalAvgPoolDiagram({ data, beforeMaps, originalImage }) {
  const canvasRef = useRef(null);
  // ... rest of the existing props and state

  const values = useMemo(() => {
    if (data && data.length > 0) return data;
    // Fallback mock data
    return Array.from({ length: 2048 }, () => Math.random() * 0.5);
  }, [data]);

  const maxVal = Math.max(...values, 0.01);

  // Render Heatmap to Canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const cols = 46;
    const rows = 45;
    const cellSize = canvas.width / cols;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    values.slice(0, 2048).forEach((v, i) => {
      const col = i % cols;
      const row = Math.floor(i / cols);
      const intensity = v / maxVal;
      
      // Vibrant Scientific Scale: Deep Blue -> Electric Cyan -> White
      const r = Math.floor(intensity * 180 + (intensity > 0.8 ? (intensity - 0.8) * 375 : 0));
      const g = Math.floor(intensity * 220 + (intensity > 0.9 ? (intensity - 0.9) * 350 : 0));
      const b = Math.floor(80 + intensity * 175);
      
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fillRect(col * cellSize, row * cellSize, cellSize - 0.4, cellSize - 0.4);
      
      // Strong Glow for high activations
      if (intensity > 0.7) {
        ctx.shadowBlur = 10;
        ctx.shadowColor = `rgba(56, 189, 248, ${intensity * 0.8})`;
        ctx.fillRect(col * cellSize, row * cellSize, cellSize - 0.4, cellSize - 0.4);
        ctx.shadowBlur = 0;
      }
    });
  }, [values, maxVal]);

  return (
    <div className="gap-analyzer">
      <div className="gap-analyzer__header">
        <h3 className="gap-analyzer__title">Pooling Analysis Matrix</h3>
        <div className="pipeline-node__tag" style={{ background: '#1e293b', color: '#38bdf8' }}>REAL-TIME DATA</div>
      </div>

      <div className="gap-analyzer__content">
        
        {/* COLUMN 1: Bar Chart */}
        <div className="gap-analyzer__panel">
          <div className="gap-analyzer__panel-title">Activations (Bins 0:64)</div>
          <div className="gap-analyzer__bar-container">
            {values.slice(0, 64).map((v, i) => (
              <motion.div 
                key={i}
                className="gap-analyzer__bar"
                initial={{ height: 0 }}
                animate={{ height: `${(v/maxVal) * 100}%` }}
                style={{ 
                  background: 'linear-gradient(to top, #1e40af, #38bdf8)',
                  boxShadow: v/maxVal > 0.8 ? '0 0 10px rgba(56,189,248,0.5)' : 'none' 
                }}
              />
            ))}
          </div>
        </div>

        {/* COLUMN 2: Canvas Heatmap */}
        <div className="gap-analyzer__panel" style={{ border: '1px solid rgba(56, 189, 248, 0.2)' }}>
          <div className="gap-analyzer__panel-title" style={{ color: '#38bdf8' }}>2048 Channel Global Matrix</div>
          <div style={{ background: '#000', padding: '6px', borderRadius: '4px', border: '1px solid #1e293b' }}>
            <canvas 
              ref={canvasRef} 
              width={300} 
              height={300} 
              style={{ width: '100%', height: 'auto', display: 'block', imageRendering: 'pixelated' }}
            />
          </div>
          <div style={{ marginTop: '10px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
             <span style={{ fontSize: '8px', color: '#475569', fontWeight: 'bold' }}>RESOLUTION: 46x45</span>
             <div className="pipeline-node__tag" style={{ fontSize: '7px', background: '#0c4a6e', color: '#7dd3fc' }}>SCALING: DYNAMIC</div>
          </div>
        </div>

        {/* COLUMN 3: Stats */}
        <div className="gap-analyzer__panel">
          <div className="gap-analyzer__stat-item">
            <div className="gap-analyzer__stat-label">Features</div>
            <div className="gap-analyzer__stat-value" style={{ color: '#38bdf8' }}>2048</div>
          </div>
          <div className="gap-analyzer__stat-item">
            <div className="gap-analyzer__stat-label">Reduction</div>
            <div className="gap-analyzer__stat-value">7x7 → 1x1</div>
          </div>
          <div style={{ marginTop: 'auto', padding: '12px', background: 'rgba(56, 189, 248, 0.08)', borderRadius: '6px', border: '1px solid rgba(56, 189, 248, 0.15)' }}>
             <p style={{ fontSize: '9px', color: '#94a3b8', lineHeight: '1.6', margin: 0 }}>
               Thuật toán nén tinh túy đặc trưng: 1 trị số đại diện cho 1 bộ lọc ngữ nghĩa.
             </p>
          </div>

        </div>

      </div>

      {/* Interactive Pooling Mechanism Demo */}
      <div style={{ marginTop: '32px', borderTop: '1px solid #1e293b', paddingTop: '20px' }}>
         <div className="gap-analyzer__panel-title" style={{ color: '#38bdf8', fontSize: '12px', marginBottom: '16px' }}>
            🛠️ Cơ Chế Phân Tích: MaxPool vs AvgPool (Comparison)
         </div>
         <PoolingMechanismDemo originalImage={originalImage} />
      </div>
    </div>
  );
}

