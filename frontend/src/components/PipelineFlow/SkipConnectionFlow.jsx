import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import './SkipConnectionFlow.css';

/**
 * SkipConnectionFlow.jsx
 * =====================
 * Trực quan hóa cơ chế Skip Connection ở cấp độ Kênh (Channels).
 * Mô phỏng luồng dữ liệu song song giữa nhánh Identity (x) và Residual (F(x)).
 */
export default function SkipConnectionFlow({ stageName, activeLevel = 0.5 }) {
  // Giả lập 10 kênh đặc trưng để minh họa
  const channels = useMemo(() => [
    { id: 1, type: 'active',   importance: 0.9, label: 'Edges' },
    { id: 2, type: 'stable',   importance: 0.2, label: 'Identity' },
    { id: 3, type: 'active',   importance: 0.7, label: 'Texture' },
    { id: 4, type: 'stable',   importance: 0.1, label: 'Identity' },
    { id: 5, type: 'ignored',  importance: 0.0, label: 'Noise' },
    { id: 6, type: 'active',   importance: 0.8, label: 'Patterns' },
    { id: 7, type: 'stable',   importance: 0.3, label: 'Identity' },
    { id: 8, type: 'active',   importance: 0.6, label: 'Shapes' },
    { id: 9, type: 'ignored',  importance: 0.0, label: 'Zero' },
    { id: 10, type: 'stable',  importance: 0.4, label: 'Identity' },
  ], []);

  return (
    <div className="skip-flow">
      <div className="skip-flow__header">
        <span className="skip-flow__title">{stageName} - Skip Connection Logic</span>
        <div className="skip-flow__legend">
           <span className="legend-item"><i className="dot dot--fx" /> F(x) Residual</span>
           <span className="legend-item"><i className="dot dot--x" /> x Identity</span>
        </div>
      </div>

      <div className="skip-flow__grid">
        {channels.map((ch, i) => (
          <div key={ch.id} className="skip-flow__channel">
            {/* Input Node */}
            <div className="node node--input" />

            {/* Path Split */}
            <div className="paths-container">
               {/* TOP PATH: F(x) - Learned Transformations */}
               <div className="path path--fx">
                  <motion.div 
                    className={`stream stream--fx ${ch.importance < 0.3 ? 'stream--dimmed' : ''}`}
                    animate={{ 
                       opacity: ch.importance > 0 ? [0.3, 1, 0.3] : 0.1,
                       scaleX: ch.importance > 0 ? [1, 1.1, 1] : 1
                    }}
                    transition={{ duration: 2, repeat: Infinity, delay: i * 0.2 }}
                  />
                  {ch.importance > 0.5 && <div className="activation-spark" />}
               </div>

               {/* BOTTOM PATH: x - Skip Connection (Always stable) */}
               <div className="path path--x">
                  <motion.div 
                    className="stream stream--x"
                    animate={{ opacity: [0.6, 0.8, 0.6] }}
                    transition={{ duration: 3, repeat: Infinity }}
                  />
               </div>
            </div>

            {/* Merge Node (Summation) */}
            <div className="node node--merge">
               <motion.span 
                 animate={{ scale: ch.importance > 0.5 ? [1, 1.3, 1] : 1 }}
                 transition={{ duration: 1, repeat: Infinity }}
               >+</motion.span>
            </div>

            {/* Output Node */}
            <div className="node node--output">
               <div 
                 className="output-glow" 
                 style={{ 
                   opacity: 0.3 + (ch.importance * 0.7),
                   background: ch.importance > 0.5 ? 'var(--accent-primary)' : 'var(--accent-secondary)'
                 }} 
               />
            </div>

            <div className="channel-label">{ch.label}</div>
          </div>
        ))}
      </div>

      <div className="skip-flow__footer">
         <p>💡 Các kênh mờ (dimmed) cho thấy khối này chọn "bỏ qua" việc học mới và giữ nguyên thông tin gốc qua đường Skip.</p>
      </div>
    </div>
  );
}
