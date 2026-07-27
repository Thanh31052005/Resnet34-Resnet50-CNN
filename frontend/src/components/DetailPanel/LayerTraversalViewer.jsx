import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './LayerTraversalViewer.css';

/**
 * Pixel-Level Traversal Viewer (Minimalist Edition)
 * Tập trung hoàn toàn vào luồng hình ảnh và animation hạt (Particles).
 */
export default function LayerTraversalViewer({ result }) {
  const [currentBlock, setCurrentBlock] = useState(0);
  const [residualData, setResidualData] = useState(null);
  const [loading, setLoading] = useState(false);

  // Danh sách các khối layer (chỉ lấy stage blocks cho tinh gọn)
  const blocks = [];
  if (result && result.layers) {
    result.layers.forEach((layer) => {
      if (layer.name.startsWith('stage')) {
        blocks.push({ id: `${layer.name}-0`, stage: layer.name, index: 0, label: `${layer.name} Block 1` });
        blocks.push({ id: `${layer.name}-1`, stage: layer.name, index: 1, label: `${layer.name} Block 2` });
      }
    });
  }

  useEffect(() => {
    if (result && blocks.length > 0 && currentBlock === 0) {
       const firstStageIndex = blocks.findIndex(b => b.id.startsWith('stage'));
       if (firstStageIndex !== -1) setCurrentBlock(firstStageIndex);
    }
  }, [result]);

  const activeBlock = blocks[currentBlock] || null;

  useEffect(() => {
    if (!activeBlock) return;
    const fetchData = async () => {
      setLoading(true);
      try {
        const response = await fetch(`http://localhost:8000/api/residual/${activeBlock.stage}/${activeBlock.index}`);
        if(response.ok) {
           const data = await response.json();
           setResidualData(data);
        }
      } catch (e) {
        setResidualData(null);
      }
      setLoading(false);
    };
    fetchData();
  }, [currentBlock]);

  if (!result || blocks.length === 0) return null;

  return (
    <div className="layer-viewer layer-viewer--minimal">
      {/* Particle Flow Layer - Animated Skip Connection */}
      <div className="layer-viewer__flow-overlay">
         {residualData && (
           <>
             {[...Array(8)].map((_, i) => (
                <motion.div 
                   key={i}
                   className="layer-viewer__particle"
                   initial={{ x: '15%', y: '50%', opacity: 0 }}
                   animate={{ 
                      x: '85%', 
                      y: ['50%', '35%', '65%', '50%'][i%4], 
                      opacity: [0, 1, 1, 0],
                      scale: [0.6, 1.2, 1, 0.6] 
                   }}
                   transition={{ 
                      duration: 2.5, 
                      repeat: Infinity, 
                      delay: i * 0.4,
                      ease: "easeInOut"
                   }}
                />
             ))}
           </>
         )}
      </div>

      <div className="layer-viewer__header">
        <h3 className="layer-viewer__title">🖼️ Pixel Flow Analysis</h3>
        <p className="layer-viewer__desc">Cơ chế truyền tin qua Identity Skip-connection.</p>
      </div>

      <div className="layer-viewer__main">
        {/* SCRUBBER NẰM DƯỚI HEADER */}
        <div className="layer-viewer__scrubber">
           <input 
              type="range" 
              min="0" max={blocks.length - 1} 
              value={currentBlock} 
              onChange={(e) => setCurrentBlock(parseInt(e.target.value))}
              className="layer-viewer__slider"
           />
           <div className="layer-viewer__active-block">{activeBlock?.label}</div>
        </div>

        <AnimatePresence mode="wait">
          {loading ? (
             <motion.div key="loader" initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}} className="layer-viewer__status">
               Loading Layer...
             </motion.div>
          ) : (residualData && (
             <motion.div 
               key="data" 
               initial={{opacity:0, scale:0.98}} 
               animate={{opacity:1, scale:1}} 
               exit={{opacity:0}} 
               className="layer-viewer__content"
             >
                <div className="layer-viewer__display">
                  {/* F(x) */}
                  <div className="layer-viewer__box">
                    <img src={`data:image/png;base64,${residualData.fx}`} className="layer-viewer__image" alt="fx" />
                    <span className="layer-viewer__tag">f(x)</span>
                  </div>

                  <div className="layer-viewer__arrow">→</div>

                  {/* F(x) + X */}
                  <div className="layer-viewer__box">
                    <img src={`data:image/png;base64,${residualData.fx_plus_x}`} className="layer-viewer__image" alt="fx+x" />
                    <span className="layer-viewer__tag">f(x) + x</span>
                  </div>
                </div>

                {/* SIGNAL INTENSITY COMPARISON */}
                <div className="layer-viewer__intensity">
                   <div className="intensity-header">
                      <span>⚡ So sánh cường độ tín hiệu</span>
                   </div>
                   <div className="intensity-bars">
                      {/* BAR 1: f(x) Residual */}
                      <div className="intensity-bar">
                         <div className="intensity-bar__label">Đặc trưng mới f(x)</div>
                         <div className="intensity-bar__track">
                            <motion.div 
                               className="intensity-bar__fill intensity-bar__fill--fx"
                               initial={{ width: 0 }}
                               animate={{ width: `${Math.min((residualData.fx_norm / residualData.output_norm) * 100, 100)}%` }}
                               transition={{ type: 'spring', stiffness: 50, damping: 15 }}
                            />
                         </div>
                         <div className="intensity-bar__value">Norm: {residualData.fx_norm?.toFixed(2)}</div>
                      </div>

                      {/* BAR 2: x Identity */}
                      <div className="intensity-bar">
                         <div className="intensity-bar__label">Thông tin gốc x</div>
                         <div className="intensity-bar__track">
                            <motion.div 
                               className="intensity-bar__fill intensity-bar__fill--x"
                               initial={{ width: 0 }}
                               animate={{ width: `${Math.min((residualData.x_norm / residualData.output_norm) * 100, 100)}%` }}
                               transition={{ type: 'spring', stiffness: 50, damping: 15 }}
                            />
                         </div>
                         <div className="intensity-bar__value">Norm: {residualData.x_norm?.toFixed(2)}</div>
                      </div>
                   </div>
                   <p className="intensity-desc">Identity $x$ chiếm tỷ trọng lớn giúp bảo toàn tín hiệu gốc khi qua lớp sâu.</p>
                </div>
             </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}
